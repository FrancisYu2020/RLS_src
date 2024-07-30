# script to plot the validation and test results discrepancy for different methods
import matplotlib.pyplot as plt
import numpy as np

# Example data
methods = [
    'Random',
    'All one',
    'Conv3D (inter val)',
    'Conv3D (intra val)',
    '3D ResNet10 (inter val)',
    '3D ResNet10 (intra val)',
    '3D ResNet18 (inter val)',
    '3D ResNet18 (intra val)',
    'Kinetics pretrained (inter val)',
    'Kinetics pretrained (intra val)',
]

# vertical group val/test compare
# validation_results = [14.52, 11.18, 8.73, 19.57, 12.73, 26.20, 9.48, 16.86]
# test_results = [5.81, 10.76, 0.93, 13.42, 7.66, 13.69, 5.06, 9.91]

# vertical group inter/intra compare
validation_results = [4.31, 4.42, 5.81, 0, 0.93, 0.00, 7.66, 0, 5.06, 0.04]
test_results = [6.71, 6.97, 5.61, 10.71, 11.50, 13.42, 8.22, 13.69, 6.18, 9.91]

# horizontal group val/test compare
# validation_results = [8.06, 4.05, 11.52, 10.00, 10.49, 6.59]
# test_results = [10.44, 3.20, 6.95, 2.93, 11.46, 2.98]

# horizontal group inter/intra compare
# validation_results = [6.40, 6.64, 10.44, 6.30, 6.95, 3.83, 11.46, 7.94] # inter-patient test results (variable name abused for convenience)
# test_results = [3.87, 3.92, 3.47, 3.20, 2.98, 2.93, 4.27, 2.98] # intra-patient test results

# Number of methods
N = len(methods)
ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

# Bar chart
rects1 = ax.bar(ind - width/2, validation_results, width, label='Inter')
rects2 = ax.bar(ind + width/2, test_results, width, label='Intra')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Methods')
ax.set_ylabel('F1 Scores')
ax.set_title('Intra vs. Inter Patient Test Results')
# ax.set_title('Validation vs. Test Results')
ax.set_ylim(0, 20)
# ax.set_ylim(0, 30)
ax.set_xticks(ind)
ax.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor', wrap=True)
ax.legend()

# Adding labels to the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
# plt.savefig('figures/val_test_horizontal_results.png')
plt.savefig('figures/intra_inter_vertical_results.png')
