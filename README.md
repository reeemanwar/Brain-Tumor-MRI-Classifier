## Brain Tumor MRI Classification
Using deep learning to classify brain MRI scans by tumor type. This project builds and compares two custom Convolutional Neural Networks (CNNs) trained to identify four categories of brain MRI images, exploring how AI can support medical image analysis.

## Project Overview
This project applies deep learning to a real-world medical imaging problem — classifying brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor.

The objective was to build a model that learns to recognize visual patterns in MRI images without any pre-programmed medical knowledge, and to evaluate whether architectural improvements to the model translate into measurable accuracy gains. Two CNN models were designed, trained, and compared to demonstrate the impact of deliberate design choices on performance.

## Business Question
Radiologists face growing imaging workloads, and delays in diagnosis can directly affect patient outcomes. This project explores how AI can serve as a decision-support tool by addressing questions such as:

* Can a neural network reliably distinguish between tumor types based solely on MRI visuals?
* Which architectural choices lead to meaningful improvements in classification accuracy?
* How does model performance vary across tumor types, and where do errors occur?

## Executive Takeaways
If developed further and validated clinically, this type of model could:

* Flag high-priority MRI scans for faster radiologist review, reducing time from scan to diagnosis.
* Support triage workflows in radiology departments managing high imaging volumes.
* Serve as a foundation for transfer learning applications across other medical imaging tasks such as chest X-rays or retinal scans.

## Methodology
The workflow followed a standard deep learning pipeline:

* Image preprocessing including resizing to 128×128 pixels and pixel normalization for consistent model input
* Data loading using PyTorch's ImageFolder and DataLoader with batch processing
* Model training over 15 epochs using the Adam optimizer and Cross-Entropy Loss

Two CNN architectures were built and evaluated:

* Basic CNN (baseline model) — 2 convolutional layers with dropout regularization
* Improved CNN (final model) — 3 convolutional layers with batch normalization and tuned dropout

The Improved CNN achieved a test accuracy of 89.81%, outperforming the baseline by 2.62 percentage points.

## Key Findings
The model comparison and per-class evaluation revealed clear patterns in classification performance:

* The Improved CNN consistently outperformed the baseline across all four tumor categories.
* No Tumor and Pituitary classes achieved the strongest classification results, where MRI visual patterns are most distinct.
* Glioma and Meningioma were the hardest classes to separate, reflecting a challenge recognized in clinical radiology where these tumor types can appear visually similar.
* Batch normalization was the single most impactful architectural change, stabilizing training and reducing overfitting.

Detailed results including the confusion matrix and per-class classification report are available in the project notebook and report.

## Repository Structure
* [Raw Data](https://github.com/reeemanwar/brain-tumor-mri-classifier/blob/main/BUS638_Project.ipynb) — all files of the raw datatset
* [Notebook](https://github.com/reeemanwar/brain-tumor-mri-classifier/blob/main/BUS638_Project.ipynb) — full model training, evaluation, and visualizations
* [Report](https://github.com/reeemanwar/brain-tumor-mri-classifier/blob/main/Report_MRI_Classification.docx) — project report with findings and business context

## Business Relevance
This project demonstrates how deep learning can be applied to high-stakes domains like healthcare, where accurate and explainable AI has direct real-world consequences. Building two models and comparing them rigorously reflects the kind of analytical discipline required in production ML environments — not just building something that works, but understanding why it works and where it falls short.

## Tools & Technologies
Python · PyTorch · torchvision · scikit-learn · Matplotlib · Deep Learning · Computer Vision · Google Colab

## Authors
Reem Anwar · Sonja Skura
Data Analytics | Machine Learning | Business Intelligence
