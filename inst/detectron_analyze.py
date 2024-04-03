# detectron_analyze.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_predictions(csv_path):
    df = pd.read_csv(csv_path)
    # Analysis code here, e.g., plot average number of objects per class
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Class Name", data=df)
    plt.title("Object Counts by Class")
    plt.show()
