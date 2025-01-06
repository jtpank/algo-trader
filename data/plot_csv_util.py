import argparse

import os

import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from datetime import datetime

def generate_plot(data_path, output_file, parameters):
    df = pd.read_csv(data_path, index_col=["Date"])

    # Determine the number of plots
    num_plots = len(parameters["plots_to_create"])

    # Create a list of subplot titles
    subplot_titles = [plot["name"] for plot in parameters["plots_to_create"]]

    # Create a list of subplot heights
    per_plot_heights = []
    for plot in parameters["plots_to_create"]:
        height = plot.get("height")
        if height is None:
            height = parameters["default_height"]
        per_plot_heights.append(height)

    total_height = sum(per_plot_heights)

    # Normalize the subplot heights so they sum to 1
    subplot_heights = [h / total_height for h in per_plot_heights]

    # Create subplots
    fig = make_subplots(
        rows=num_plots,
        cols=1,
        subplot_titles=subplot_titles,
        row_heights=subplot_heights,
        vertical_spacing=parameters["spacing_factor"] / num_plots,
    )

    fig.update_layout(
        title=" ".join(
            [parameters["plot_name"], "Data: " + os.path.basename(data_path)]
        )
    )

    start_dt = datetime.strptime(parameters.get("start_time"), "%Y-%m-%d")
    # if start is None:
    #     start = 0
    start = parameters.get("start_time")
    end_dt = datetime.strptime(parameters.get("end_time"), "%Y-%m-%d")
    # if end is None:
    #     end = len(df)
    end = parameters.get("end_time")

    subsample_factor = parameters.get("subsample")
    if subsample_factor is None:
        subsample_factor = 1

    for i, plot in enumerate(parameters["plots_to_create"], start=1):
        for signal in plot["signals"]:
            # if isinstance(signal_item, str):
            #     signal = signal_item
            #     display_signal_name = signal_item
            # elif isinstance(signal_item, dict):
            #     signal, display_signal_name = next(iter(signal_item.items()))
            # else:
            #     raise ValueError("Invalid signal format in configuration.")

            # Check if the signal exists in the dataframe
            if signal not in df.columns:
                print(f"Warning: Signal {signal} not found in dataframe.")
                continue

            x_vals = df.loc[start:end].index.to_list()
            y_vals = df.loc[start:end][signal].to_list()

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=signal,
                ),
                row=i,
                col=1,
            )

        # Update the layout of the i-th subplot
        fig.update_yaxes(title_text=plot["y_label"], row=i, col=1)
        fig.update_xaxes(
            title_text=plot.get("x_label"),
            row=i,
            col=1,
            title_standoff=5,
        )

    # Link the x-axes of all subplots
    for idx in range(2, num_plots + 1):
        fig.update_xaxes(matches="x", row=idx, col=1)

    # Update the layout of the figure
    fig.update_layout(
        title="<br>".join(
            [parameters["plot_name"], "Data: " + os.path.basename(data_path)]
        ),
        title_x=0,
        height=total_height,
        hovermode="x unified",
    )
    fig.layout.hoverlabel.namelength = -1

    # Check the file extension
    extension = os.path.splitext(output_file)[1]
    if extension != ".html":
        print(f"Warning: The file {output_file} does not have the .html extension.")

    if parameters.get("plot_show", False):
        fig.show()
    if parameters.get("plot_save", True):
        fig.write_html(file=output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate plots from CSV data.")

    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=False,
        default="shares_held.html",
        type=str,
        help="Path to the output file.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=False,
        default="plot_config.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()

    # Get the script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Determine config file path
    config_path = (
        args.config_file
        if args.config_file
        else os.path.join(script_dir, "plot_config.yaml")
    )

    # Load parameters
    with open(config_path, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)["plot_types"]["time_series_2d"]
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    # Determine data_path
    data_path = args.input_file if args.input_file else parameters["data_path"]

    generate_plot(data_path, args.output_file, parameters)

if __name__ == "__main__":
    main()
