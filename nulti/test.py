import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
import math 
import data
  


# Example precision-recall pairs
precision = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
recall = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]

# Plot precision-recall curve
plt.plot(recall, precision, marker='o', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Fill area under the curve
for i in range(len(precision) - 1):
    plt.fill_between(recall[i:i+2], precision[i:i+2], alpha=0.3)

# Calculate AP
delta_recall = [recall[i + 1] - recall[i] for i in range(len(recall) - 1)]
print(f"delta recall {delta_recall}")
average_precision = sum((0.5 * (precision[i] + precision[i+1])) * delta_recall[i] for i in range(len(precision) - 1))

# Display AP
plt.text(0.5, 0.2, f'AP = {average_precision:.2f}', fontsize=12, ha='center')

plt.legend()
plt.grid(True)
plt.show()