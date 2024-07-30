# script to plot the validation and test results discrepancy for different methods
import matplotlib.pyplot as plt
import numpy as np

# Example data
methods = [
#     'Random',
#     'All one',
    'Conv3D (inter val)',
    'Conv3D (intra val)',
    '3D ResNet10 (inter val)',
    '3D ResNet10 (intra val)',
    '3D ResNet18 (inter val)',
    '3D ResNet18 (intra val)',
    'Kinetics pretrained (inter val)',
    'Kinetics pretrained (intra val)',
]

# # horizontal group validation results
# validation_results = [
#     [0.84, 16.56, 11.29, 13.47, 21.43],
#     [6.14, 5.93, 6.31, 7.79, 4.69],
#     [0.59, 13.70, 22.34, 22.25, 19.32],
#     [10.65, 8.16, 10.71, 11.85, 19.82],
#     [1.83, 12.90, 20.88, 25.56, 20.65],
#     [9.52, 5.34, 4.62, 7.36, 7.53],
#     [5.57, 24.44, 16.07, 18.51, 19.49],
#     [7.97, 11.56, 11.48, 19.52, 11.56]
# ]

# Vertical group validation results
validation_results = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    []
]

means = [np.array(arr).mean() for arr in validation_results]
stds = [np.array(arr).std() for arr in validation_results]
# vertical group val/test compare
# validation_results = [14.52, 11.18, 8.73, 19.57, 12.73, 26.20, 9.48, 16.86]
# test_results = [5.81, 10.76, 0.93, 13.42, 7.66, 13.69, 5.06, 9.91]

# vertical group inter/intra compare
# validation_results = [4.31, 4.42, 5.81, 0, 0.93, 0.00, 7.66, 0, 5.06, 0.04]
# test_results = [6.71, 6.97, 5.61, 10.71, 11.50, 13.42, 8.22, 13.69, 6.18, 9.91]

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
rects1 = ax.bar(ind - width/2, means, width, yerr=stds)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Methods')
ax.set_ylabel('F1 Scores')
ax.set_title('Validation Score Variance on 5 Val Sets for Different Methods')
# ax.set_title('Validation vs. Test Results')
# ax.set_ylim(0, 20)
# ax.set_ylim(0, 30)
ax.set_xticks(ind)
ax.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor', wrap=True)

# Adding labels to the bars
def autolabel(rects, stds):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect, std in zip(rects, stds):
        height = rect.get_height()
        ax.annotate(f'{height:.2f}±{std:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, stds)

fig.tight_layout()
plt.savefig('figures/val_test_horizontal_variance.png')
# plt.savefig('figures/intra_inter_vertical_results.png')
