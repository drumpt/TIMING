import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

model = ""
data = "hmm"
experiment_name = "_modify"

# Define the base file name pattern
file_pattern = "{data}_{cv}_results{experiment_name}.csv"

# Assuming you have 5 folds (CV = 0 to 4) and a specific topk value (e.g., 0.2)
cv_folds = 5
top_value = 100

# Define the columns in the CSV file
columns = [
    "Seed", "CV", "Baseline", "Explainer", "Lambda_1", "Lambda_2",
    "top50_cum", "top0.1_cum", "AUCC", "aup", "aur", "information", "entropy", "roc_auc", "auprc"
]

# Initialize a dictionary to store dataframes for each CV fold
dataframes = []


# Loop through each CV fold and read the corresponding CSV file
for cv in range(cv_folds):
    file_name = file_pattern.format(data=data,cv=cv, experiment_name=experiment_name)
    df = pd.read_csv(file_name, header=None, names=columns)
    dataframes.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# Group by the relevant columns (excluding Seed, CV, and Lambda columns)
grouped = combined_df.groupby(["Baseline", "Explainer"])

# Define a function to calculate standard error
def standard_error(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))

# Calculate mean and standard error across the CV folds
result = grouped.agg(
    mean_aup=("aup", np.mean),
    std_error_aup=("aup", standard_error),
    mean_aur=("aur", np.mean),
    std_error_aur=("aur", standard_error),
    mean_information=("information", np.mean),
    std_error_information=("information", standard_error),
    mean_entropy=("entropy", np.mean),
    std_error_entropy=("entropy", standard_error),
    mean_top50=("top50_cum", np.mean),
    std_error_top50=("top50_cum", standard_error),
    mean_topK=("top0.1_cum", np.mean),
    std_error_topK=("top0.1_cum", standard_error),
    mean_AUCC=("AUCC", np.mean),
    std_error_AUCC=("AUCC", standard_error),
)

# for mimic, only when my code is wrong...
# for col in result.columns:
#     if "top" in col:
#         result[col] = result[col] / 2

result = result.applymap(lambda x: f"{x:.3f}")

# Define the output directory and file path
output_dir = Path(f"./results/{data}")
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
output_file = output_dir / f"{model}_{experiment_name}.csv"

# Save the result to the CSV file
result.to_csv(output_file)

print(f"Results saved to {output_file}")