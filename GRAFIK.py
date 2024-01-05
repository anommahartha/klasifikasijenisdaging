import matplotlib.pyplot as plt
import pandas as pd

# Data for the experiments
data_specific = {
    "No": [1, 2, 3],
    "Ukuran Citra (Pixel)": ["50 X 50", "50 X 50", "50 X 50"],
    "Epoch": [50, 50, 50],
    "Batch size": [32, 64, 128],
    "Accuracy (%)": [74.70, 74.70, 70.91],
    "Precision (%)": [84.07, 82.36, 80.02],
    "Recall (%)": [72.73, 72.81, 68.95],
    "F1 Score (%)": [73.06, 72.40, 67.20]
}

df_specific = pd.DataFrame(data_specific)

# Set different colors for each metric
colors = {
    "Accuracy (%)": 'b',
    "Precision (%)": 'g',
    "Recall (%)": 'r',
    "F1 Score (%)": 'm'
}

# Creating a figure for the plot
plt.figure(figsize=(10, 6))

# Plotting each metric and adding annotations
for metric, color in colors.items():
    plt.plot(df_specific["Batch size"], df_specific[metric], color=color, marker='o', linestyle='-', label=metric)
    for x, y in zip(df_specific["Batch size"], df_specific[metric]):
        plt.text(x, y, f"{y:.2f}%", color=color, fontsize=10, ha='center', va='bottom')

# Adding labels and title
plt.xlabel("Batch Size")
plt.ylabel("Percentage")
plt.title("CNN Performance Metrics for 50 X 50 Pixel Image Size")
plt.legend(loc='best')
plt.xticks(df_specific["Batch size"])
plt.grid(True)

# Adjust layout to ensure everything fits
plt.tight_layout()

# Show the plot
plt.show()
