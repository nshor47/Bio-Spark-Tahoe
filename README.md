Bio-Spark-Tahoe

📊 Group Project Milestone 4: 2nd Model and Final Submission

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

Full Write Up:

Tahoe-100M- A Machine Learning Treasure Trove

Introduction
This project leverages gene expression data from the Tahoe-100M dataset to predict transcriptomic responses to drug perturbations in glioblastoma-derived human cell lines. We selected this project because of the significant potential it holds for personalized medicine, particularly in cancer therapeutics. The application of machine learning models to predict how cancer cells respond to various drug treatments is not only scientifically intriguing but also practically valuable. By accurately predicting cell responses, we can facilitate more targeted drug development and effective personalized treatment plans, potentially leading to better patient outcomes. The broader impact of developing robust predictive models lies in accelerating biomedical research, reducing the time and cost associated with drug discovery, and enhancing the precision of clinical interventions.

1.Methods
Preprocessing

Gene Expression Preprocessing from the Tahoe-100M Dataset

To facilitate downstream machine learning on transcriptomic responses to drug perturbations, we developed a scalable preprocessing pipeline to extract, filter, and transform gene expression summaries from the Tahoe-100M dataset,a large-scale HuggingFace dataset of RNA-seq perturbations organized by cell types and drug classes.

Dataset Sampling and Filtering Strategy

We aimed to use a subset of the Tahoe-100M dataset corresponding to seven glioblastoma-derived human cell lines: A-172 (CVCL_0131), U-251MG (CVCL_0021), U-87MG (CVCL_0022), U-343MG (CVCL_S471), D-54MG (CVCL_5735), H4 (CVCL_1239), and BT048 (CVCL_RU29). To efficiently manage resource usage in limited environments (e.g., Google Colab and local machines), we used batch-based streaming using datasets.load_dataset(..., split=f"train[{i}:{i+BATCH_SIZE}]") with batch sizes of 500 records and optional slicing ranges (SLICE_START, SLICE_END). Each sample was filtered to retain only entries with a valid cell_line_id from the defined set and a non-empty expressions list containing at least 10 numeric values.

Summarization and Feature Extraction

For each valid sample, we computed descriptive summary statistics of the gene expression array: mean (expr_mean), standard deviation (expr_std), skewness (expr_skew), and length (expr_len). Additional metadata fields including drug and moa-fine (mechanism of action) were also retained. Summarized entries were appended line-by-line to a local JSONL file (summaries.jsonl), enabling resumption and incremental progress tracking via a checkpoint file (progress_marker.txt).

Preprocessing Pipeline and Feature Engineering

After data collection, the JSONL file was parsed into a Pandas dataframe and subjected to a multi-stage preprocessing pipeline. Numerical features (expr_mean, expr_std, expr_skew, expr_len) were imputed using a mean strategy for any missing values (via SimpleImputer), scaled to zero-mean, unit-variance using z-score normalization (StandardScaler), and transformed via second-order polynomial expansion to introduce pairwise interaction terms

Categorical variables (cell_line_id, drug, moa-fine) were encoded using one-hot encoding with handling for unseen categories. All transformer objects (scaler, encoder, imputer, and polynomial expansion) were serialized and saved for downstream reuse (in the models/directory). The final feature matrix was saved as a CSV (final_preprocessed.csv), containing the complete transformed feature set suitable for machine learning workflows.

Design Considerations and Extensibility

To support robustness across long-running sessions and large-scale data volumes, the pipeline includes automatic caching via Hugging Face’s datasets backend, local batching to control memory usage, and a resume-safe mechanism that avoids reprocessing completed batches. While this study utilized only summary-level statistics of expression arrays, the framework is extensible to include full vectorized expression profiles or embedded representations using TSNE/UMAP-based reductions.
Data Exploration
We began our analysis by exploring the gene expression data across samples. A histogram of all positive expression values(Figure 1), plotted on a logarithmic x-axis, showed a heavily right-skewed distribution. The majority of expression values clustered tightly between 1 and 10, with very few extending beyond 100. This pattern is consistent with biological gene expression datasets, where most genes are either inactive or expressed at low levels. The use of a log-scale effectively visualized the skew and highlighted the small subset of highly expressed genes.

![Hist of Gene Expression Tahoe](https://github.com/user-attachments/assets/44ef1f97-d776-4cb6-ada7-5b13e10c337e)

Figure 1.  The histogram displays the distribution of positive gene expression values on a logarithmic x-axis. The majority of expression values are tightly concentrated between 1 and 10, with a sharp drop-off in frequency as expression levels increase beyond 10. Very few genes show expression values above 10, and almost none beyond 100. The log-scale x-axis highlights the heavy skew toward low expression values, typical in gene expression datasets where most genes are either lowly expressed or inactive.



To examine the breadth of gene expression per sample, we computed a new column(Figure 2)representing the number of expressed genes (non-null entries) per sample. This distribution was also right-skewed, with most samples containing between 800 to 1,200 genes and a peak around 1,000. A long tail extended toward 3,500 genes, though only a few samples reached this range. This variability suggests that while most samples share a similar range of active genes, there are some with unusually high gene counts that may represent distinct biological states or data artifacts.

![Hist of Gene 2 Tahoe](https://github.com/user-attachments/assets/5a70373b-2f07-4261-ae44-3501b22dd74b)

	Figure 2.  The histogram displays the distribution of positive gene expression values on a logarithmic x-axis. The majority of expression values are tightly concentrated between 1 and 10, with a sharp drop-off in frequency as expression levels increase beyond 10. Very few genes show expression values above 10, and almost none beyond 100. The log-scale x-axis highlights the heavy skew toward low expression values, typical in gene expression datasets where most genes are either lowly expressed or inactive.
We also assessed categorical feature distributions to better understand the dataset’s composition. A bar chart(Figure 3) of the top 10 most frequent drugs revealed a fairly even distribution among them, with the most common drug, (R)-Verapamil (hydrochloride), occurring about 100 times. The other top drugs, such as Aliskiren and Palmatine (chloride), followed closely behind with frequencies in the 80–95 range, suggesting a relatively balanced representation among the top treatments.

![Tahoe 3 Fig](https://github.com/user-attachments/assets/8af4f35b-415c-42c2-9c0f-b94b4d8f147d)

Figure 3. The bar chart displays the top 10 most frequent drugs in the dataset. (R)-Verapamil (hydrochloride) appears most frequently with approximately 100 counts, closely followed by Aliskiren with around 95 counts. The remaining drugs show a gradual decrease in frequency: Palmatine (chloride) and Quinestrol at roughly 90 counts each, DMSO_TF at about 85 counts, followed by Isocorydine, Megestrol, Pasireotide (acetate), Terfenadine, and Temsirolimus, with the latter having approximately 80 counts. The visualization illustrates the relative prevalence of these drugs in the analyzed sample, with a relatively small difference (about 20 counts) between the most and least frequent drugs in the top 10.
However, cell line frequency was far more imbalanced. CVCL_0131 appeared in roughly 4,000 samples, four times more often than the next most common cell line, CVCL_1239, with around 1,000 samples. This significant imbalance could introduce bias into downstream models if not accounted for. A similar issue was observed in the mechanism of action (MoA) data. The “Unclear” MoA category dominated with over 3,200 occurrences—more than ten times that of the next highest category, DNA synthesis/repair inhibitors. Other MoAs appeared far less frequently, ranging from 50 to 300 samples. This high prevalence of ambiguous labels may impact the interpretability of any models relying on MoA as a target or feature.
These findings highlight key patterns and imbalances in the data that guided preprocessing, modeling choices, and evaluation strategies in later stages of the analysis.

Model 1: Ridge Regression 
The first model implemented was a Ridge Regression model, selected for its ability to handle multicollinearity and regularize feature weights. The target variable was delta_mean, representing changes in gene expression, and the input features included one-hot encoded drug and cell line identifiers. The dataset was split into training and test sets using an 80/20 split to assess generalization. The model was trained with a regularization parameter (alpha) of 1.0, a standard starting point to penalize overly large coefficients.

Model 2: Random Forest Classification 
The second model applied was a Random Forest Classifier, aimed at predicting cell line identity from a combination of drug encoding features and expression-related metrics (mean_expression, delta_mean, and expressions_length). Labels were derived by finding the active cell line (from one-hot encoded columns) for each row, then encoding them as integers using LabelEncoder. This model type was chosen due to its ability to capture non-linear relationships and handle mixed feature types without requiring normalization or strong assumptions about feature distributions.

2.Results
Model 1: Ridge Regression Evaluation
Upon evaluation, the model yielded a training Mean Squared Error (MSE) of 0.9863 and a test MSE of 0.9631, indicating consistent performance between training and unseen data. Similarly, the coefficient of determination (R²) was low on both sets: 0.0183 for training and 0.0185 for testing. 

![Rigde Regression Tahoe100M](https://github.com/user-attachments/assets/58916838-f626-4e5f-850b-e9084213f5a2)

Model 2: Random Forest Classification Evaluation
The dataset was again split into 80% training and 20% testing. After fitting the model with 100 decision trees, it achieved perfect performance on the training set with an accuracy of 1.0000. On the test set, accuracy remained extremely high at 0.9974, with the classification report showing near-perfect precision, recall, and F1-scores for both classes (CVCL_0131 and CVCL_1239). 

![RFC for Tahoe 100M](https://github.com/user-attachments/assets/7a84255c-b7a0-40cf-abe5-319d6ed2b56e)


3.Discussion
Model 1: Ridge Regression 
	
	Evaluation yielded a training Mean Squared Error (MSE) of 0.9863 and a test MSE of 0.9631, indicating consistent performance between training and unseen data. Similarly, the coefficient of determination (R²) was low on both sets: 0.0183 for training and 0.0185 for testing. These results suggest the model generalizes well (since there’s minimal performance drop between training and test), but it fails to capture much of the variance in the target variable. In other words, while the model is not overfitting, it also doesn’t perform well due to the weak predictive power of the selected features in explaining gene expression changes. This indicates that either more informative features are required or that the relationship between inputs and target is too complex for a linear model to capture effectively.

Model 2: Random Forest Classification 

	After fitting the model with 100 decision trees, it achieved perfect performance on the training set with an accuracy of 1.0000. On the test set, accuracy remained extremely high at 0.9974, with the classification report showing near-perfect precision, recall, and F1-scores for both classes (CVCL_0131 and CVCL_1239). The minimal performance drop between training and test sets suggests very strong generalization despite the perfect training fit. The classifier’s performance indicates that the input features are highly predictive of cell line identity, likely due to strong, distinct signal patterns between drug effects and cell line behavior. However, despite the minimal performance drop, the perfect training accuracy raises a mild overfitting concern. Incorporating cross-validation can help with the model’s robustness on unseen data. 
This stark contrast with Model 1 highlights the difference in difficulty between the two tasks. While regression on delta_mean appears to be a challenging problem with weakly informative features (or potentially noisy labels), classification of cell lines using those same or similar features performs exceedingly well. This suggests that the dataset may be more amenable to categorical prediction tasks, or that the biology encoded in the features more directly separates cell lines than it predicts specific expression shifts.
Although the Random Forest model resulted in an extremely small gap of 0.26%, there is still room to improve performance and robustness when the model is deployed on completely new data. One way to ensure this is to cross validate with different Random Forest configurations to confirm consistent performance. By experimenting with the number of trees, max_depth. and min_samples_split, we can prevent overfitting and improve its generalization on unseen data. These changes can help balance the bias-variance tradeoff. 
Ideas for Future Investigation
It is also worth investigating simpler baseline models, such as Logistic Regression and Support Vector Machine (SVM). By doing so, we can understand the minimum complexity needed for classification and improve runtime. If our data’s features have clear linear separability, classification may be simpler than what our Random Forest model suggests. In other words, it can help us assess whether the task requires the complexity of ensemble methods. Lastly, the SVM model is effective with high dimensional data and is able to find complex boundaries through kernel methods. Even if the data is not linearly separable, the use of kernel functions allows SVM to find non-linear boundaries. SVMs require less hyperparameter tuning than tree models, and they are a strong candidate for classification performance and assessing the need for complex models like Random Forest.

Conclusion

	Reflecting on this project, several aspects stand out clearly, both in terms of accomplishments and potential areas for further improvement. The use of the Tahoe-100M dataset offered an exceptional foundation for exploring drug perturbation responses in glioblastoma cell lines, proving to be a powerful asset for machine learning applications in biomedical research. The developed preprocessing pipeline efficiently handled data complexity, ensuring scalability and reproducibility. Nevertheless, there are clear areas where future efforts can enhance the pipeline's effectiveness and generalization capability.
The Ridge Regression model's limited predictive performance underscores the inherent complexity of predicting precise gene expression changes from drug perturbations. A significant takeaway is the need for richer, possibly more biologically-informed features, or alternative modeling approaches capable of capturing complex nonlinear interactions more effectively. Future iterations should incorporate additional biological context or feature engineering techniques, perhaps including direct vectorized gene expression data or embeddings generated through dimensionality reduction methods such as TSNE or UMAP.
Conversely, the exceptional performance of the Random Forest classifier in predicting cell line identities suggests that the data inherently contains strong categorical signals, highlighting the viability of classification-based approaches. Despite its near-perfect accuracy, this model raises concerns about potential overfitting, reinforcing the importance of rigorous validation strategies, particularly cross-validation, to ensure robustness when faced with unseen data.
Moving forward, exploring simpler baseline models such as Logistic Regression or SVMs would provide valuable insights into the necessary model complexity and potentially reduce computational overhead. SVM models, in particular, merit investigation given their known strength in handling high-dimensional and complex datasets with fewer hyperparameters to tune.
Ultimately, while our initial exploration has laid a solid groundwork, further iterative development, richer feature integration, and rigorous model evaluation strategies are essential next steps. Embracing these improvements will significantly enhance the reliability and applicability of predictive models in personalized medicine, particularly in advancing targeted therapeutics for glioblastoma and other cancers.

Collaboration
1.James Conde- Organizer, writer, submitter
2. Paige Saengsouvanna- Model Execution and Analysis
3.Julia Haynes- Preprocessing and initial modeling
4. Bryan Duoto- Initial Idea/vision, data pre-processing
5.Nick Shor - Model Execution and Analysis



