
# PTO-BM: Primary Tumor Origin Classifier in Brain Metastases

This repository contains the Python code used for the radiomics-based classification of primary tumor origin in patients with brain metastases. This project was conducted at the ImagineQuant Lab, Department of Radiology & Biomedical Imaging, Yale School of Medicine.

## 🧠 Project Overview

This script implements a baseline machine learning classifier using radiomic features from cranial magnetic resonance imaging (contast-enhanced T1-weighted sequence, FLAIR sequence) to predict the primary origin of brain metastases. It includes model training using `XGBoost`, evaluation using ROC curves, and hyperparameter tuning.

## 📄 Associated Publication

_Jekel, L._ et al. (2025). "Classifying brain metastases by primary tumor etiology on MRI: addressing multiplicity of lesions, heterogeneity of metastases, and burdens of institutional datasets" (to be published)

## 🚀 Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pto-bm-classifier.git
cd pto-bm-classifier
```

2. Create environment and install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Python file:
```bash
python FinalCode_1.py
```

> 📌 Note: Input data paths need to be configured inside the script.

## 📦 Dependencies

See `requirements.txt` for the full list.

## 📬 Citation

To be published. 


