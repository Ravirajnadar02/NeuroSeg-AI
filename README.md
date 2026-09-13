NeuroSeg AI is an AI-powered brain tumor segmentation system designed to assist in identifying and outlining tumor regions in MRI scans. The project uses a SegFormer-B2 semantic segmentation model trained on multi-channel brain MRI data to classify each pixel into three categories: background, necrotic/non-enhancing tumor core (NCR/NET), and edema + enhancing tumor (ED/ET combined).

The system accepts an .h5 MRI slice containing four MRI channels, applies preprocessing and normalization, and uses the trained deep-learning model to generate a tumor segmentation mask. Test-time augmentation using horizontal and vertical flips is used during inference to improve prediction robustness. The resulting segmentation is overlaid on the MRI image so that users can visually compare the predicted tumor region with the ground-truth mask when it is available.

On the held-out test set, the model achieved a 78.16% tumor-region Dice score, with 83.64% recall and 73.35% precision, evaluated across 8,680 slices from 56 volumes. We also tested the deployed application on unseen MRI slices to verify that the model can produce meaningful segmentations outside the development examples.

The project is intended as a research and educational prototype, not as a clinical diagnostic system. Our goal is to demonstrate how modern deep-learning segmentation techniques can help analyze brain MRI scans and provide an accessible interface for experimenting with automated tumor-region segmentation

🤖 AI Tools Disclosure:
ChatGPT was used as an AI-assisted development tool during the project for code development, debugging, explanation of machine-learning concepts, troubleshooting deployment issues, improving the Streamlit interface, and preparing project documentation. The model architecture, training process, dataset preparation, experimentation, and evaluation were developed and performed as part of the project team’s work.


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
