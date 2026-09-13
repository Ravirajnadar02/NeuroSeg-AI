NeuroSeg AI is an AI-powered brain tumor segmentation system designed to assist in identifying and outlining tumor regions in MRI scans. The project uses a SegFormer-B2 semantic segmentation model trained on multi-channel brain MRI data to classify each pixel into three categories: background, necrotic/non-enhancing tumor core (NCR/NET), and edema + enhancing tumor (ED/ET combined).

The system accepts an .h5 MRI slice containing four MRI channels, applies preprocessing and normalization, and uses the trained deep-learning model to generate a tumor segmentation mask. Test-time augmentation using horizontal and vertical flips is used during inference to improve prediction robustness. The resulting segmentation is overlaid on the MRI image so that users can visually compare the predicted tumor region with the ground-truth mask when it is available.

On the held-out test set, the model achieved a 78.16% tumor-region Dice score, with 83.64% recall and 73.35% precision, evaluated across 8,680 slices from 56 volumes. We also tested the deployed application on unseen MRI slices to verify that the model can produce meaningful segmentations outside the development examples.

The project is intended as a research and educational prototype, not as a clinical diagnostic system. Our goal is to demonstrate how modern deep-learning segmentation techniques can help analyze brain MRI scans and provide an accessible interface for experimenting with automated tumor-region segmentation

🤖 AI Tools Disclosure:
ChatGPT was used as an AI-assisted development tool during the project for code development, debugging, explanation of machine-learning concepts, troubleshooting deployment issues, improving the Streamlit interface, and preparing project documentation. The model architecture, training process, dataset preparation, experimentation, and evaluation were developed and performed as part of the project team’s work.

How it works
How it works: Upload a 4-channel BraTS-style .h5 MRI slice → select the MRI channel for visualization → run SegFormer-B2 segmentation → view the predicted tumor regions → compare against ground truth when available → inspect Grad-CAM to understand the regions influencing the prediction → review Dice and class-level results.


🧪 Demo Results
<img width="1311" height="554" alt="image" src="https://github.com/user-attachments/assets/ffef0f17-64e0-493f-87ec-02489fc85b89" />
<img width="1282" height="526" alt="image" src="https://github.com/user-attachments/assets/9a9942ea-6c4a-4cc9-996c-60c8526220ae" />


Example 1 — Unseen BraTS 2020 slice

Metric	Result
NCR/NET Dice	92.3%
ED/ET Dice	100%

Prediction closely matches the ground-truth tumor segmentation, while Grad-CAM highlights the region contributing to the model's prediction.

Example 2 — Unseen BraTS 2020 slice

Metric	Result
NCR/NET Dice	90.1%
ED/ET Dice	100%

These are individual demonstration-slice results, not the overall test-set performance.

⚠️ Current Limitations
1. Limited dataset diversity:
The model is trained and evaluated on BraTS 2020 data, so performance on MRI scans from other hospitals, scanners, populations, or datasets may differ.

2. 2D slice-based segmentation:
The current system processes individual 2D MRI slices, rather than analyzing the complete 3D brain volume. This can lose spatial information between adjacent slices.

3. Limited tumor classes
Your current model predicts 3 classes: background, NCR/NET, and combined ED/ET. It does not separately distinguish every possible tumor sub-region.

4. No clinical validation
The reported 78.16% tumor-region Dice is a research/test-set result, not clinical validation. The system has not been evaluated by radiologists or in a clinical workflow.

5. Input format dependency
The demo expects a specific 4-channel .h5 format. Standard MRI files such as DICOM or NIfTI cannot currently be uploaded directly.

6. CPU inference can be slower
The application can run without a GPU, but inference may be slower on CPU than on a CUDA-enabled GPU.

7. Grad-CAM is an explanation, not a guarantee
Grad-CAM shows regions that contributed to the model's prediction; it does not prove that the highlighted region is medically correct.

8. Limited external testing
Our additional demonstrations show strong results on unseen BraTS slices, but these are still from the BraTS dataset family. We have not yet performed a true external-dataset validation.
