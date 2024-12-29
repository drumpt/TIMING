# %%
import pandas as pd


def get_result(csv_dir):
    try:
        # Read log file
        df = pd.read_csv(csv_dir)  # replace with your actual log file path

        # Group by level_0 and calculate means
        grouped_means = (
            df.groupby("level_0")
            .agg(
                {
                    "auc_drop": "mean",
                    "avg_pred_diff": "mean",
                    "avg_masked_count": "mean",
                }
            )
            .round(5)
        )

        # Convert to more readable format
        summary_df = pd.DataFrame(
            {
                "level_0": grouped_means.index,
                "avg_auc_drop": grouped_means["auc_drop"],
                "avg_pred_diff": grouped_means["avg_pred_diff"],
                "avg_masked_count": grouped_means["avg_masked_count"],
            }
        )

        # If you want to also calculate standard deviations:
        grouped_stds = (
            df.groupby("level_0")
            .agg({"auc_drop": "std", "avg_pred_diff": "std", "avg_masked_count": "std"})
            .round(5)
        )

        # Save to CSV if needed
        summary_df.to_csv("summary_means.csv", index=False)
        grouped_stds.to_csv("summary_stds.csv")
        pd.set_option("display.float_format", lambda x: "%.5f" % x)
    except:
        default_rows = [
            "top50_std_mean",
            "top50_end_mean",
            "top50_mam_mean",
            "globaltop5_std_mean",
            "globaltop5_end_mean",
            "globaltop5_mam_mean",
            "bal50_std_mean",
        ]

        df = pd.DataFrame(
            {
                "level_0": default_rows * 5,  # 5 CVs
                "cv": [i for i in range(5) for _ in range(len(default_rows))],
                "auc_drop": 0.0,
                "avg_pred_diff": 0.0,
                "avg_masked_count": 0.0,
            }
        )

        # Group by level_0 and calculate means
        grouped_means = (
            df.groupby("level_0")
            .agg(
                {
                    "auc_drop": "mean",
                    "avg_pred_diff": "mean",
                    "avg_masked_count": "mean",
                }
            )
            .round(5)
        )

        # Create summary dataframe
        summary_df = pd.DataFrame(
            {
                "level_0": grouped_means.index,
                "avg_auc_drop": grouped_means["auc_drop"],
                "avg_pred_diff": grouped_means["avg_pred_diff"],
                "avg_masked_count": grouped_means["avg_masked_count"],
            }
        )

        # Calculate standard deviations
        grouped_stds = (
            df.groupby("level_0")
            .agg({"auc_drop": "std", "avg_pred_diff": "std", "avg_masked_count": "std"})
            .round(5)
        )
    return summary_df


def get_metrics_table(basedir, dataset, modeltype, modelformat, explainer_list):
    """
    Creates a table with explainers as rows and metrics as columns for a specific modeltype
    """
    # Initialize empty lists to store results
    results = []

    # Collect results for each explainer
    for explainer in explainer_list:
        csv_dir = f"{basedir}/{modelformat}/{dataset}/{dataset}_{explainer}_{modeltype}_all_masking.csv"
        import os
        print(f"{csv_dir=}")
        print(f"{os.path.exists(csv_dir)=}")
        result_df = get_result(csv_dir)

        # Extract metrics for each evaluation type
        metrics = {}
        metrics["explainer"] = explainer

        # Add metrics for each evaluation type
        for idx, row in result_df.iterrows():
            eval_type = idx.split("_mean")[0]  # Remove '_mean' suffix
            metrics[f"{eval_type}_auc_drop"] = row["avg_auc_drop"]
            metrics[f"{eval_type}_pred_diff"] = row["avg_pred_diff"]
            metrics[f"{eval_type}_masked_count"] = row["avg_masked_count"]

        results.append(metrics)

    # Convert to DataFrame
    table_df = pd.DataFrame(results)

    # Set explainer as index
    table_df.set_index("explainer", inplace=True)

    # Define column order
    column_order = [
        "top50_mam_auc_drop",
        "top50_mam_pred_diff",
        "top50_mam_masked_count",
        "globaltop5_mam_auc_drop",
        "globaltop5_mam_pred_diff",
        "globaltop5_mam_masked_count",
        "top50_end_auc_drop",
        "top50_end_pred_diff",
        "top50_end_masked_count",
        "globaltop5_end_auc_drop",
        "globaltop5_end_pred_diff",
        "globaltop5_end_masked_count",
        "top50_std_auc_drop",
        "top50_std_pred_diff",
        "top50_std_masked_count",
        "globaltop5_std_auc_drop",
        "globaltop5_std_pred_diff",
        "globaltop5_std_masked_count",
        "bal50_std_auc_drop",
        "bal50_std_pred_diff",
        "bal50_std_masked_count",
    ]

    # Reorder columns
    table_df = table_df[column_order]

    return table_df


# %%
basedir = "../output"
dataset = "mimic"

# modeltype_list = ["gru", "mtand"]
modeltype_list = ["seft"]
# modelformat_list = ["gru1layer", "MTAND"]
modelformat_list = ["SEFT"]
# explainer_list = [
#     "deeplift",
#     "gradientshap",
#     "ig",
#     "fo",
#     "afo",
#     "fit",
#     "dynamask",
#     "winit",
#     "winitset",
#     "dynamaskset",
# ]
explainer_list = [
    "deeplift", "gradientshap", "ig", "fo", "afo", "fit", "dynamask", "winit", "winitsetzero", "winitsetzerolong", "winitsetcf", "fitsetzero", "fitsetcf", "fozero"
]

for modeltype, modelformat in zip(modeltype_list, modelformat_list):
    print(f"\nResults for {modeltype.upper()}:")
    print("-" * 80)

    table = get_metrics_table(basedir, dataset, modeltype, modelformat, explainer_list)
    table.to_csv(f"result_{modeltype}.csv")

    # Format the table for better readability
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)

    print(table)
    print("\n")
# %%
