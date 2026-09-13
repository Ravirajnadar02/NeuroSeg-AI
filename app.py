"""
NeuroSeg AI — Streamlit interactive demo

Upload a single BraTS-style 4-channel MRI slice (.h5, with an "image"
dataset of shape (H, W, 4) and optionally a "mask" dataset) and get back
the model's tumor segmentation prediction.

Research demo only — not a diagnostic tool.
"""

import io
import os

import h5py
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# =====================================================================
# Page configuration
# =====================================================================

st.set_page_config(
    page_title="NeuroSeg AI",
    page_icon="🧠",
    layout="wide",
)


# =====================================================================
# Config — must match the trained run
# =====================================================================

MODEL_NAME = "nvidia/mit-b2"
NUM_CLASSES = 3
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "best_neuroseg_segformer_b2_4ch.pth",
)

# Verify these against your dataset's real label documentation before
# presenting results publicly.
CLASS_NAMES = {
    0: "background",
    1: "necrotic_non_enhancing_tumor_core_NCR_NET",
    2: "edema_and_enhancing_tumor_ED_EC",
}

CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (255, 60, 60, 140),
    2: (255, 220, 40, 140),
}

# Held-out test-set results from the completed training notebook.
# These are dataset-level metrics, NOT the score of the uploaded slice.
TEST_METRICS = {
    "slices": 8680,
    "volumes": 56,
    "foreground_dice": 0.7816,
    "foreground_iou": 0.6414,
    "foreground_recall": 0.8364,
    "foreground_precision": 0.7335,
    "class1_dice": 0.7858,
    "class2_dice": 0.7695,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# =====================================================================
# Preprocessing — must match training-time preprocessing exactly
# =====================================================================

def prepare_4ch_image(image):
    """
    image: numpy array, shape (H, W, C) with C in {1, 3, 4}, raw MRI
    intensities.

    Returns: float32 array, shape (4, H, W), per-channel z-scored on
    nonzero voxels, clipped to [-3, 3], and rescaled to [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = image[..., None]

    c = image.shape[-1]

    if c == 1:
        image = np.repeat(image, 4, axis=-1)
    elif c == 3:
        image = np.concatenate([image, image[..., :1]], axis=-1)
    elif c != 4:
        raise ValueError(f"Unexpected number of channels: {c}")

    image = np.transpose(image, (2, 0, 1))

    normed = np.zeros_like(image, dtype=np.float32)

    for ch in range(image.shape[0]):
        chan = image[ch]
        nonzero = chan[chan != 0]

        if nonzero.size > 0:
            mean, std = nonzero.mean(), nonzero.std() + 1e-6
        else:
            mean, std = 0.0, 1.0

        z = (chan - mean) / std
        z = np.clip(z, -3, 3)
        normed[ch] = (z + 3) / 6.0

    return normed.astype(np.float32)


def mask_to_class_indices(mask):
    """Convert one-hot or integer-labelled masks to (H, W) class indices."""
    mask = np.asarray(mask)

    if mask.ndim == 3:
        return np.argmax(mask, axis=-1).astype(np.int64)

    return mask.astype(np.int64)


# =====================================================================
# Model loading
# =====================================================================

@st.cache_resource(show_spinner=False)
def build_model():
    config = SegformerConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_CLASSES
    config.id2label = CLASS_NAMES
    config.label2id = {v: k for k, v in CLASS_NAMES.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Change the first patch embedding to accept 4 MRI channels.
    old_proj = model.segformer.stages[0].patch_embeddings.proj

    model.segformer.stages[0].patch_embeddings.proj = nn.Conv2d(
        4,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
    )

    model.config.num_channels = 4

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Place best_neuroseg_segformer_b2_4ch.pth beside app.py "
            "or set CHECKPOINT_PATH."
        )

    ckpt = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# =====================================================================
# Inference — flip-based test-time augmentation
# =====================================================================

@torch.no_grad()
def predict_probs_tta(model, images):
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=USE_AMP,
    ):
        probs = torch.softmax(model(images).logits, dim=1)

        flipped_h = torch.flip(images, dims=[3])
        probs = probs + torch.flip(
            torch.softmax(model(flipped_h).logits, dim=1),
            dims=[3],
        )

        flipped_v = torch.flip(images, dims=[2])
        probs = probs + torch.flip(
            torch.softmax(model(flipped_v).logits, dim=1),
            dims=[2],
        )

    return probs / 3.0


def dice_score(pred, target, cls, eps=1e-6):
    p = pred == cls
    t = target == cls
    inter = np.logical_and(p, t).sum()
    return (2 * inter + eps) / (p.sum() + t.sum() + eps)


# =====================================================================
# Visualization helpers
# =====================================================================

def to_display_gray(channel_2d):
    ch = channel_2d.astype(np.float32)
    ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
    gray = (ch * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def overlay_mask(gray_rgb, mask, alpha_scale=1.0):
    base = Image.fromarray(gray_rgb, mode="RGB").convert("RGBA")
    overlay_np = np.zeros(
        (gray_rgb.shape[0], gray_rgb.shape[1], 4),
        dtype=np.uint8,
    )

    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue

        r, g, b, a = color
        overlay_np[mask == cls] = (
            r,
            g,
            b,
            int(a * alpha_scale),
        )

    overlay = Image.fromarray(overlay_np, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# =====================================================================
# Main prediction function
# =====================================================================

def run_inference(uploaded_file, display_channel):
    if uploaded_file is None:
        return None, None, "Upload a .h5 file to get started."

    file_bytes = uploaded_file.getvalue()

    with h5py.File(io.BytesIO(file_bytes), "r") as h:
        if "image" not in h:
            raise ValueError(
                "This file has no 'image' dataset — is it the right format?"
            )

        raw_image = h["image"][:]
        raw_mask = h["mask"][:] if "mask" in h else None

    if raw_image.size == 0 or not np.isfinite(raw_image).all():
        raise ValueError("The MRI image is empty or contains invalid values.")

    if np.count_nonzero(raw_image) == 0:
        raise ValueError(
            "This MRI slice contains no non-zero image data. "
            "Please upload a valid MRI slice."
        )

    image_4ch = prepare_4ch_image(raw_image)
    tensor = (
        torch.from_numpy(image_4ch)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    model = build_model()

    probs = predict_probs_tta(model, tensor)

    target_size = (image_4ch.shape[1], image_4ch.shape[2])

    if probs.shape[-2:] != target_size:
        probs = F.interpolate(
            probs,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy()

    ch_idx = min(int(display_channel), image_4ch.shape[0] - 1)
    gray_rgb = to_display_gray(image_4ch[ch_idx])

    pred_overlay = overlay_mask(gray_rgb, pred_mask)

    summary_lines = [
        f"**Predicted class breakdown** "
        f"(MRI channel {ch_idx} shown as background):"
    ]

    total_px = pred_mask.size

    for cls, name in CLASS_NAMES.items():
        pct = (pred_mask == cls).sum() / total_px * 100
        summary_lines.append(f"- {name}: {pct:.2f}% of pixels")

    gt_overlay = None

    if raw_mask is not None:
        gt_indices = mask_to_class_indices(raw_mask)
        gt_overlay = overlay_mask(gray_rgb, gt_indices)

        summary_lines.append(
            "\n**Dice vs. ground-truth mask in this file:**"
        )

        for cls, name in CLASS_NAMES.items():
            if cls == 0:
                continue

            d = dice_score(pred_mask, gt_indices, cls)
            summary_lines.append(
                f"- {name}: Dice = {d:.3f}"
            )
    else:
        summary_lines.append(
            "\n_No ground-truth mask found in this file — "
            "showing prediction only._"
        )

    return pred_overlay, gt_overlay, "\n".join(summary_lines)


# =====================================================================
# Streamlit UI
# =====================================================================

st.title("🧠 NeuroSeg AI — Brain Tumor Segmentation")
st.markdown(
    "SegFormer-B2 fine-tuned on 4-channel MRI slices "
    "(T1 / T1ce / T2 / FLAIR-style inputs)."
)
st.warning("Research demo only — not a diagnostic tool.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tumor-region Dice", "78.16%")
m2.metric("Recall", "83.64%")
m3.metric("Precision", "73.35%")
m4.metric("Held-out slices", "8,680")

st.caption(
    "Model performance above is from the held-out test set "
    "(56 volumes). It is not the score of the uploaded image."
)

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload MRI slice (.h5)",
        type=["h5"],
        help=(
            "The file should contain an 'image' dataset with shape "
            "(H, W, 4), and may optionally contain a 'mask' dataset."
        ),
    )

    display_channel = st.selectbox(
        "MRI channel to display as background",
        options=["0", "1", "2", "3"],
        index=3,
    )

    run_btn = st.button(
        "Run segmentation",
        type="primary",
        use_container_width=True,
    )

with col2:
    if run_btn:
        if uploaded_file is None:
            st.warning("Please upload a .h5 MRI slice first.")
        else:
            try:
                with st.spinner(
                    "Running SegFormer-B2 segmentation..."
                ):
                    pred_overlay, gt_overlay, summary = run_inference(
                        uploaded_file,
                        display_channel,
                    )

                st.success("Segmentation completed.")

                image_col1, image_col2 = st.columns(2)

                with image_col1:
                    st.image(
                        pred_overlay,
                        caption="Predicted segmentation",
                        use_container_width=True,
                    )

                with image_col2:
                    if gt_overlay is not None:
                        st.image(
                            gt_overlay,
                            caption="Ground truth (if available)",
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "No ground-truth mask was found in this file."
                        )

                st.markdown(summary)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

st.divider()

st.markdown(
    """
**Legend:** 🔴 red = class 1, 🟡 yellow = class 2,
transparent = background.

**Before public submission:** verify `CLASS_NAMES` against your
dataset's real label documentation.
"""
)
