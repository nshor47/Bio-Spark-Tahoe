Bio-Spark-Tahoe

📊 Group Project Milestone 3: Preprocessing, Model Training & Evaluation

How to Run youself-  Simply download the Bio_Spark_major_preprocessing.ipynb and run it in google collab.

Project Overview

This milestone builds upon our previous data exploration and preprocessing work on the TAHOE-100M gene expression dataset. We completed major preprocessing tasks, trained our initial predictive model, evaluated its performance, and identified next steps for improvement.  Bio_Spark_major_preprocessing is the file on which the information covered was excuted from.   

📁 Dataset Information

Name: TAHOE-100M

Format: JSONL / CSV

Source: [Huggingface Dataset Link
](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
Size: 100 million samples

Content:

Drug perturbation identifiers

Gene expression values (~978 landmark genes)

Cell line identifiers

Metadata (e.g., SMILES, concentrations)

📈 Preprocessing Completed

Filtering and Imputation:

Retained records with valid cell_id.

Imputed missing gene expression values with mean/median imputation.

Scaling and Encoding:

Scaled numerical gene expression features using standardization (mean=0, variance=1).

Encoded categorical drug metadata using one-hot encoding.

Feature Expansion:

Generated additional features via polynomial (degree=2) and log transformations of selected features.

🚀 Model Training

First Model: Random Forest Regression

Trained on the processed dataset with an 80-20 train-test split.

Hyperparameters tuned via grid search.

📊 Model Evaluation

Training Error: RMSE = 0.23

Test Error: RMSE = 0.35

Fitting Analysis:

The model shows signs of slight overfitting, evidenced by lower training error compared to test error.

🔍 Model Performance Insights

Fit in Underfitting/Overfitting Graph:

Slightly in the overfitting region, indicating the model complexity could be slightly reduced or regularization increased.

🛠️ Next Steps

Explore simpler models like Ridge or Lasso regression to mitigate overfitting.

Consider additional dimensionality reduction techniques (e.g., PCA) to improve model generalization.

📂 Files and Notebooks

clean_Final_Edits_Tahoe_100M_Stream_and_Filter_cellid.ipynb: Contains previous filtering and slicing procedures.

Bio_Spark_major_preprocessing.ipynb: Includes completed major preprocessing, model training, and evaluation steps.

📌 Example Predictions

Ground truth vs. predicted gene expression provided in Bio_Spark_major_preprocessing.ipynb for train, validation, and test datasets.

📝 Conclusion

Our initial Random Forest model performs reasonably but exhibits slight overfitting. Future improvements include regularization and exploring simpler linear models to enhance predictive generalization.
