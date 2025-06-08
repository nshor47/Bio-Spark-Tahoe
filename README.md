Bio-Spark-Tahoe

📊 Group Project Milestone 3: Preprocessing, Model Training & Evaluation

How to Run

Simply download the Bio_Spark_major_preprocessing.ipynb notebook and run it in Google Colab.

Project Overview

This milestone builds upon our previous data exploration and preprocessing work on the TAHOE-100M gene expression dataset. We completed extensive preprocessing, trained multiple predictive models (Ridge Regression and Random Forest Classification), evaluated their performances, and identified future improvement opportunities. All operations are performed in the Bio_Spark_major_preprocessing.ipynb notebook.

📁 Dataset Information

Name: TAHOE-100M

Format: JSONL / CSV

Source: Huggingface Dataset

Size: 100 million samples

Content:

Drug perturbation identifiers

Gene expression values (~978 landmark genes)

Cell line identifiers

Metadata (e.g., SMILES, concentrations)

📈 Preprocessing Completed

Filtering and Imputation

Retained records with valid cell_line_id and minimum gene expression data.

Imputed missing numerical features (expression summary statistics) using mean imputation.

Scaling and Encoding

Scaled numerical gene expression summary features (mean, std, skewness, length) using standardization (mean=0, variance=1).

Encoded categorical metadata (drug, cell line, mechanism of action) using one-hot encoding.

Feature Expansion

Applied second-order polynomial transformations to numerical features.

🚀 Model Training

Model 1: Ridge Regression

Target: Gene expression change (delta_mean).

Split: 80% train, 20% test.

Hyperparameters: Regularization parameter (alpha=1.0).

Model 2: Random Forest Classification

Target: Cell line identity.

Split: 80% train, 20% test.

Hyperparameters: Number of trees (100).

📊 Model Evaluation

Ridge Regression

Training MSE: 0.9863

Test MSE: 0.9631

R²: Low (~0.018), indicating limited predictive performance.

Random Forest Classification

Training Accuracy: 100%

Test Accuracy: 99.74%

Indicates strong predictive power with slight concern about potential overfitting.

🔍 Model Performance Insights

Ridge Regression struggled due to limited feature informativeness for gene expression prediction.

Random Forest excelled in classification, suggesting categorical prediction is highly feasible.

Cross-validation recommended to confirm model robustness.

🛠️ Next Steps

Explore simpler models (Logistic Regression, Support Vector Machines) for classification.

Include additional biologically-informed features or embedding methods (TSNE/UMAP) to enhance predictive performance.

Implement rigorous cross-validation for model validation.

📂 Files and Notebooks

clean_Final_Edits_Tahoe_100M_Stream_and_Filter_cellid.ipynb: Filtering and slicing procedures.

Bio_Spark_major_preprocessing.ipynb: Preprocessing pipeline, model training, and evaluation steps.

📌 Example Predictions

Ground truth vs. predicted values for gene expression (delta_mean) and cell line classification provided in Bio_Spark_major_preprocessing.ipynb.

📝 Conclusion

Our analysis identified clear paths for model improvement. While Ridge Regression demonstrated limited predictive power, Random Forest classification provided excellent performance with minor overfitting concerns. Future improvements include deeper feature engineering, additional embedding techniques, and rigorous validation methods to ensure robust, generalizable models.



