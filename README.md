
# PTO-BM: Primary Tumor Origin Classifier in Brain Metastases

This repository contains the Python code used for the radiomics-based classification of primary tumor origin in patients with brain metastases. This project was conducted at the ImagineQuant Lab, Department of Radiology & Biomedical Imaging, Yale School of Medicine.

## 🧠 Project Overview

This script implements a baseline machine learning classifier using radiomic features from cranial magnetic resonance imaging (contast-enhanced T1-weighted sequence, FLAIR sequence) to predict the primary origin of brain metastases. It includes model training using `XGBoost` with nested cross-validation, feature selection and model evaluation using ROC curves. It also includes radiomic feature aggregation and model training with additional clinical information and qualitative information derived from imaging.  

## 📄 Associated Publication

_Jekel, L._ et al. (2025). "Classifying brain metastases by primary tumor etiology on MRI: addressing multiplicity of lesions, heterogeneity of metastases, and burdens of institutional datasets" (to be published)

## 🚀 Getting Started


1. Create environment and install dependencies:
pip install -r requirements.txt

2. Extract radiomic features from your preprocessed segmentation masks using PyRadiomics with the `params.yaml` file

3. Perform radiomic feature aggregation according to the comnputations described in Chang et al. (https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-BrainMetsRadiomics)

4. Link clinical features and the semantic, qualitative imaging features to the radiomics frame with `link_features.py`.

5. Run the classifier using `classification.py`.

> 📌 Note: Input data paths need to be configured inside the script.

## 📦 Dependencies

See `requirements.txt` for the full list.

## 📬 Citation

To be published. 


