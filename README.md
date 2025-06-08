# Bio-Spark-Tahoe

📊 Group Project Part 4: 2nd Model and Final Submission
🔍 Project Overview
This project is part of our data science group work focused on exploring and preprocessing a large-scale gene expression dataset. Specifically, we are working with the TAHOE-100M dataset, which contains transcriptional response profiles from various drug perturbations in different cell types.

📁 Dataset Information
Name: TAHOE-100M

Format: JSONL / CSV

Source: https://huggingface.co/datasets/tahoebio/Tahoe-100M

Size: 100 million samples

Content:

Drug perturbation identifiers

Gene expression values

Cell line identifiers

Metadata fields (e.g., SMILES, concentrations)

📈 Data Exploration Summary
Initial Filtering: Only records containing valid cell_id values were retained.

Observations: After filtering, we observed a significantly reduced but still large dataset with millions of usable records.

Features:

High-dimensional gene expression arrays (approximately 978 landmark genes)

Drug metadata fields such as drug_name, smiles, concentration, and time

Distributions: Distributions of gene expression values and concentration levels were analyzed using histograms and boxplots (see notebook).

Missing Data: Rows with missing cell_id or malformed records were discarded during the streaming process.

🧪 Preprocessing Strategy
The dataset preprocessing involved the following:

Streaming the Dataset: Due to memory limitations, we implemented a streaming method to load and process the dataset in chunks.

Filtering: We selected rows with a valid cell_id and discarded broken or irrelevant entries.

Slicing: The cleaned dataset was split into 5,000 partitioned slices for parallel and incremental downstream processing.

This strategy enabled efficient computation and modular analysis in distributed environments such as Google Colab or Spark-based platforms.

📂 Files in This Repository
clean_Final_Edits_Tahoe_100M_Stream_and_Filter_cellid.ipynb: Jupyter Notebook containing all filtering, streaming, slicing, and visualizations.

