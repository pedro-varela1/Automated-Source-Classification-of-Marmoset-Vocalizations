import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the data
data = {
    'Individual': ['Blue2A', 'Blue2C', 'Blue3A', 'Blue3B', 'Blue3C', 
              'Pink2C', 'Pink3A', 'Pink3B', 'Pink3C'],
    'Precision': [0.8723, 0.8673, 0.9178, 0.8867, 0.9077, 
                  0.9259, 0.8984, 0.8729, 0.9207],
    'Recall': [0.8652, 0.8815, 0.9259, 0.9104, 0.9037, 
               0.9252, 0.8970, 0.8904, 0.8681],
    'F1-Score': [0.8687, 0.8744, 0.9218, 0.8984, 0.9057, 
                 0.9255, 0.8977, 0.8816, 0.8936],
    'AUC': [0.9889, 0.9905, 0.9961, 0.9929, 0.9936,
            0.9966, 0.9925, 0.9914, 0.9944],
    'Total Samples': [1350] * 9
}

df = pd.DataFrame(data)

# Create figure and axis
plt.figure(figsize=(12, 6))
plt.axis('off')

# Create table
df_formatted = df.copy()
numeric_cols = ['Precision', 'Recall', 'F1-Score', 'AUC']
df_formatted[numeric_cols] = df_formatted[numeric_cols].map(lambda x: f"{x:.4f}")

table = plt.table(cellText=df_formatted.values, 
                  colLabels=df.columns, 
                  cellLoc='center', 
                  loc='center', 
                  colColours=['lightgray']*len(df.columns))

# Customize table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.5)  # Make rows taller

# Add title
plt.title('Test Performance Metrics by Individual', fontsize=16, fontweight='bold', pad=0)

# Save the figure
plt.tight_layout()
plt.savefig('results_table.png', dpi=300, bbox_inches='tight')
plt.close()