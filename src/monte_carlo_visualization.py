import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_optical_misalignment_distribution(optical_misalignment_filepath):
    # Read the CSV file, skipping comment lines
    df = pd.read_csv(optical_misalignment_filepath, comment='#')
    df['yaw'] = df['yaw'] * 180 / np.pi
    df['pitch'] = df['pitch'] * 180 / np.pi
    df['roll'] = df['roll'] * 180 / np.pi
    print(df.head())
    print(df.columns)
    print(df)

    unique_panels = df['Panel ID'].unique()
    target_metrics = ['radial_extention', 'radial_shift', 'elevation_shift', 'yaw', 'pitch', 'roll']
    target_metric_ranges = [(df[metric].min(axis=0), df[metric].max(axis=0)) for metric in target_metrics]

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=len(target_metrics), ncols=len(unique_panels), figsize=(9, 9))

    for i, metric in enumerate(target_metrics):
        for j, panel in enumerate(unique_panels):
            # Filter the DataFrame for the current panel
            panel_df = df[df['Panel ID'] == panel]

            # Plot the distribution of the current metric for the current panel, with axes swapped
            sns.violinplot(y=panel_df[metric], ax=ax[i, j], orient='v', inner='quartile', split=True)
            ax[i, j].set_title(f'{panel} - {metric}', fontsize=10)
            ax[i, j].set_ylim(target_metric_ranges[i][0], target_metric_ranges[i][1])
            if j == 0:
                ax[i, j].set_ylabel(f"{metric} ({'cm' if i < 3 else 'deg'})", fontsize=10)
            else:
                ax[i, j].set_ylabel('')
                ax[i, j].set_yticklabels([])
            if i == len(target_metrics) - 1:
                ax[i, j].set_xlabel('Frequency', fontsize=10)
            else:
                ax[i, j].set_xlabel('')

    # Optionally, set a main title for the figure
    fig.suptitle('Distribution of Optical Misalignment in Arm Configuration', fontsize=16)

    # Show the plot
    plt.show()

def plot_compare_optical_misalignment_distribution(longarm_optical_misalignment_filepath, shortarm_optical_misalignment_filepath):
    # Read the CSV file, skipping comment lines
    longarm_df = pd.read_csv(longarm_optical_misalignment_filepath, comment='#')
    longarm_df['yaw'] = longarm_df['yaw'] * 180 / np.pi
    longarm_df['pitch'] = longarm_df['pitch'] * 180 / np.pi
    longarm_df['roll'] = longarm_df['roll'] * 180 / np.pi
    shortarm_df = pd.read_csv(shortarm_optical_misalignment_filepath, comment='#')
    shortarm_df['yaw'] = shortarm_df['yaw'] * 180 / np.pi
    shortarm_df['pitch'] = shortarm_df['pitch'] * 180 / np.pi
    shortarm_df['roll'] = shortarm_df['roll'] * 180 / np.pi

    unique_panels = longarm_df['Panel ID'].unique()
    target_metrics = ['radial_extention', 'radial_shift', 'elevation_shift', 'yaw', 'pitch', 'roll']
    target_metric_ranges = [(min(longarm_df[metric].min(axis=0), shortarm_df[metric].min(axis=0)), max(longarm_df[metric].max(axis=0), shortarm_df[metric].max(axis=0))) for metric in target_metrics]

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=len(target_metrics), ncols=int(2*len(unique_panels)), figsize=(18, 9))

    for i, metric in enumerate(target_metrics):
        for j, panel in enumerate(unique_panels):
            # Filter the DataFrame for the current panel
            panel_df = longarm_df[longarm_df['Panel ID'] == panel]

            # Plot the distribution of the current metric for the current panel, with axes swapped
            sns.violinplot(y=panel_df[metric], ax=ax[i, j], orient='v', inner='quartile', split=True)
            ax[i, j].set_title(f'{panel} - {metric}', fontsize=8)
            ax[i, j].set_ylim(target_metric_ranges[i][0], target_metric_ranges[i][1])
            if i == len(target_metrics) - 1:
                ax[i, j].set_xlabel('Frequency', fontsize=10)
            else:
                ax[i, j].set_xlabel('')
            if j == 0:
                ax[i, j].set_ylabel(f"{metric} ({'cm' if i < 3 else 'deg'})", fontsize=10)
            else:
                ax[i, j].set_ylabel('')
                ax[i, j].set_yticklabels([])

    for p, metric in enumerate(target_metrics):
        for q, panel in enumerate(unique_panels):
            # Filter the DataFrame for the current panel
            panel_df = shortarm_df[shortarm_df['Panel ID'] == panel]
            q_shift = q + len(unique_panels)

            # Plot the distribution of the current metric for the current panel, with axes swapped
            sns.violinplot(y=panel_df[metric], ax=ax[p, q_shift], orient='v', inner='quartile', split=True, color='skyblue')
            ax[p, q_shift].set_title(f'{panel} - {metric}', fontsize=8)
            ax[p, q_shift].set_ylim(target_metric_ranges[p][0], target_metric_ranges[p][1])
            ax[p, q_shift].set_yticklabels([])
            if p == len(target_metrics) - 1:
                ax[p, q_shift].set_xlabel('Frequency', fontsize=10)
            else:
                ax[p, q_shift].set_xlabel('')
            ax[p, q_shift].set_ylabel('')
            ax[p, q_shift].set_yticklabels([])

    # Optionally, set a main title for the figure
    fig.suptitle('LONG ARMS     - Comparative Distribution of Optical Misalignment in Configuration -     SHORT ARMS', fontsize=14)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Set the file path for the optical misalignment data
    longarm_optical_misalignment_filepath = r"C:\Users\thetk\OneDrive\Documents\GitHub\NASA-Model-T-KC-Simulation\src\monte_carlo_sims\side_arm_monte_carlo\linked_panel_monte_carlo_optical_results_20250516_024415.csv"
    plot_optical_misalignment_distribution(longarm_optical_misalignment_filepath)

    shortarm_optical_misalignment_filepath = r"C:\Users\thetk\OneDrive\Documents\GitHub\NASA-Model-T-KC-Simulation\src\monte_carlo_sims\short_arm_monte_carlo\linked_panel_monte_carlo_optical_results_20250516_025100.csv"
    plot_optical_misalignment_distribution(shortarm_optical_misalignment_filepath)

    plot_compare_optical_misalignment_distribution(longarm_optical_misalignment_filepath, shortarm_optical_misalignment_filepath)