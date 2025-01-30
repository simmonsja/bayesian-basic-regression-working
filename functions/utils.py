import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

def rescale_target(data, scale_min_vals, scale_max_vals, target_name = "dW", target_appends = ["", "_pred", "_paper"]):
    """
    Rescale target variable using the scale values.
    """
    data = data.assign(
        scale_min = data["location"].map(scale_min_vals[target_name]),
        scale_max = data["location"].map(scale_max_vals[target_name])
    )
    for col in [target_name + _ for _ in target_appends]:
        data[col] = data[col] * (data["scale_max"] - data["scale_min"]) + data["scale_min"]

    data = data.drop(columns=["scale_min", "scale_max"])
    return data

def calculate_r2_rmse(data, target_name = "dW", target_appends = ["_pred", "_paper"]):
    """
    Calculate R2 score for the target variable.
    """
    r2_scores = {
        "r2" + _ : pd.DataFrame() for _ in target_appends
    }
    rmse_scores = {
        "rmse" + _ : pd.DataFrame() for _ in target_appends
    }

    for this_loc in data["location"].unique():
        this_df = data.loc[data["location"] == this_loc]

        for this_append in target_appends:
            r2_scores["r2" + this_append].loc[this_loc, "r2"] = r2_score(this_df[target_name], this_df[target_name + this_append])
            rmse_scores["rmse" + this_append].loc[this_loc, "rmse"] = root_mean_squared_error(this_df[target_name], this_df[target_name + this_append])
    return r2_scores, rmse_scores