import seaborn as sns
import matplotlib.pyplot as plt

def plot_scatter_predictions(data, r2_scores, rmse_scores, target_name="dW", target_append="", title=""):
    """
    Plot scatter plot of predictions.
    """
    # plot a scatter that is facetted by location
    g = sns.FacetGrid(data, col="location", col_wrap=3, height=3)
    g.map(sns.scatterplot, target_name, f"{target_name}_pred")
    g.set_axis_labels(f"Observed {target_name}", f"Predicted {target_name}")
    # red dashed 1:1 line
    for ax in g.axes:
        ax.plot([data[target_name].min(), data[target_name].max()], [data[target_name].min(), data[target_name].max()], color="red", linestyle="--", alpha=0.5)
    # print the R2 values in top left of each plot per location
    for ax, r2 in zip(g.axes, r2_scores["r2" + target_append]["r2"].values):
        ax.text(0.05, 0.95, f"R2: {r2:.2f}", transform=ax.transAxes, verticalalignment="top", fontsize=10)
    # print the RMSE values in top left underneath the R2 values
    for ax, rmse in zip(g.axes, rmse_scores["rmse"+target_append]["rmse"].values):
        ax.text(0.05, 0.85, f"RMSE: {rmse:.2f}", transform=ax.transAxes, verticalalignment="top", fontsize=10)

    fig = ax.get_figure()
    if title:
        fig.suptitle(title, y=1.05)

    plt.show()

def fix_forest_plots(ax):
    # loop through each y axis tick label and make it subscript the location
    ax.set_yticklabels([_.get_text().replace("[", "$_{[").replace("]", "]}$") for _ in ax.get_yticklabels()])
    # set x axis gridlines as grey
    ax.grid(axis="x", color="lightgrey")
    return ax