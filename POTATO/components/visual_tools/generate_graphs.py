import csv
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Dict

# set current dir to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

testdata = [
    {"month": "January", "value": 100},
    {"month": "February", "value": 120},
    {"month": "March", "value": 90},
    {"month": "April", "value": 110},
    {"month": "May", "value": 130},
    {"month": "June", "value": 115},
    {"month": "July", "value": 140},
    {"month": "August", "value": 40},
    {"month": "September", "value": 135},
    {"month": "October", "value": 16},
    {"month": "November", "value": 170},
    {"month": "December", "value": 180},
]

available_graph_types = [
    'line', 'bar', 'scatter', 'pie', 'histogram', 'box', 'area', 'heatmap',
    'radar', 'donut', 'funnel', 'gauge', 'waterfall', 'sunburst', 'violin',
    'density', 'step', 'errorbar', 'pairplot'
]
recommended_graph_types = [
    'line', 'bar', 'scatter', 'pie', 'histogram', 'box', 'area', 'heatmap', 'radar', 'donut', 'funnel', 'errorbar'
]

# Draw any graph based on given data and parameters, including type. type can be 'line', 'bar', 'scatter', 'pie', etc.
def draw_any_graph_type(
    data: List[Dict],
    x_key: str,
    y_key: str,
    graph_type: str,
    title: str = "Graph",
    x_label: str = "X-axis",
    y_label: str = "Y-axis",
    output_dir: str = "graphs_output",
    output_name: str = "graph.png"
):
    try:
        x = [item[x_key] for item in data]
        y = [item[y_key] for item in data]
    except KeyError as e:
        raise ValueError(f"Key error: {e}. Available keys: {list(data[0].keys()) if data else []}")

    plt.figure(figsize=(10, 6))

    # Line plot
    if graph_type == 'line':
        plt.plot(x, y, marker='o')
    # Bar plot
    elif graph_type == 'bar':
        plt.bar(x, y)
    # Scatter plot
    elif graph_type == 'scatter':
        plt.scatter(x, y)
    # Pie chart (no axis labels)
    elif graph_type == 'pie':
        plt.pie(y, labels=x, autopct='%1.1f%%')
        plt.xlabel("")
        plt.ylabel("")
    # Histogram (y is the data, x is ignored)
    elif graph_type == 'histogram':
        plt.hist(y, bins='auto')
        plt.xlabel(y_label)
        plt.ylabel("Frequency")
    # Box plot (y is the data, x is ignored)
    elif graph_type == 'box':
        plt.boxplot(y)
        plt.xlabel("")
    # Area chart
    elif graph_type == 'area':
        plt.fill_between(x, y, alpha=0.5)
    # Heatmap (y must be 2D or will reshape)
    elif graph_type == 'waterfall':
        cumulative = np.cumsum(y)
        plt.bar(x, cumulative)
    elif graph_type == 'sunburst':
        wedges, texts = plt.pie(y, labels=x, wedgeprops=dict(width=0.5))
        plt.xlabel("")
        plt.ylabel("")
    elif graph_type == 'heatmap':
        arr = np.array(y)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        plt.imshow(arr, cmap='hot', interpolation='nearest')
        plt.colorbar()
    # Radar chart (x are categories, y are values)
    elif graph_type == 'radar':
        angles = np.linspace(0, 2 * np.pi, len(x), endpoint=False).tolist()
        y_radar = y + [y[0]]
        angles += [angles[0]]
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, y_radar, color='red', alpha=0.25)
        ax.plot(angles, y_radar, color='red', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(x)
        plt.xlabel("")
        plt.ylabel("")
    # Donut chart
    elif graph_type == 'donut':
        wedges, texts, autotexts = plt.pie(y, labels=x, autopct='%1.1f%%', wedgeprops=dict(width=0.3))
        plt.setp(autotexts, size=10, weight="bold")
        plt.xlabel("")
        plt.ylabel("")
    # Funnel chart (horizontal bar, sorted)
    elif graph_type == 'funnel':
        sorted_pairs = sorted(zip(y, x), reverse=True)
        y_sorted, x_sorted = zip(*sorted_pairs)
        plt.barh(x_sorted, y_sorted)
        plt.xlabel(y_label)
        plt.ylabel(x_label)
    # Gauge chart (single value)
    elif graph_type == 'gauge':
        fig, ax = plt.subplots()
        ax.barh([0], [y[0]], color='blue')
        ax.set_xlim(0, max(y)*1.2)
        ax.set_yticks([])
        ax.set_title(title)
    # Violin plot
    elif graph_type == 'violin':
        plt.violinplot(y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    # Density plot (KDE)
    elif graph_type == 'density':
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(y)
            y_vals = np.linspace(min(y), max(y), 100)
            plt.plot(y_vals, kde(y_vals))
            plt.xlabel(y_label)
            plt.ylabel("Density")
        except ImportError:
            raise ImportError("Density plot requires scipy.")
    # Step plot
    elif graph_type == 'step':
        plt.step(x, y, where='mid')
    # Errorbar plot (requires yerr in data)
    elif graph_type == 'errorbar':
        yerr = [item.get('yerr', 0) for item in data]
        plt.errorbar(x, y, yerr=yerr, fmt='o')
    # Pairplot (scatter matrix, needs pandas)
    elif graph_type == 'pairplot':
        try:
            import pandas as pd
            import seaborn as sns
            df = pd.DataFrame(data)
            sns.pairplot(df)
        except ImportError:
            raise ImportError("Pairplot requires pandas and seaborn.")
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    # Only set axis labels if not already set (for pie, radar, etc.)
    if plt.gca().get_xlabel() == "":
        plt.xlabel(x_label)
    if plt.gca().get_ylabel() == "":
        plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_name), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs("graphs_output", exist_ok=True)
    # test various graph types with testdata
    for gtype in recommended_graph_types:
        draw_any_graph_type(
            data=testdata,
            x_key="month",
            y_key="value",
            graph_type=gtype,
            title=f"Monthly Values - {gtype.capitalize()} Plot",
            x_label="Month",
            y_label="Value",
            output_name=f"{gtype}_plot.png"
        )

    # Errorbar plot test (add yerr to testdata)
    testdata_errorbar = [dict(item, yerr=10) for item in testdata]
    draw_any_graph_type(
        data=testdata_errorbar,
        x_key="month",
        y_key="value",
        graph_type="errorbar",
        title="Monthly Values - Errorbar Plot",
        x_label="Month",
        y_label="Value",
        output_name="errorbar_plot.png"
    )

    # Pairplot test (requires pandas and seaborn)
    try:
        draw_any_graph_type(
            data=testdata,
            x_key="month",
            y_key="value",
            graph_type="pairplot",
            title="Monthly Values - Pairplot",
            x_label="Month",
            y_label="Value",
            output_name="pairplot.png"
        )
    except ImportError:
        print("Pairplot requires pandas and seaborn. Skipping pairplot test.")

    print("Graphs generated in 'graphs_output' directory.")