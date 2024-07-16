import numpy as np
from PIL import Image

data = np.load('data/patient15-04-12-2024-relabeled/win40_tal_val_data_full.npy').reshape(-1, 48, 22)[0, ...]
data[data > 255] = 255
data = data.astype(np.uint8)
print(data)

img = Image.fromarray(data).convert('RGB')
colored_array = np.array(img)
print(((colored_array[:, :, 0] - colored_array[:, :, 1])**2).sum(), ((colored_array[:, :, 0] - colored_array[:, :, 2])**2).sum())
img.save('test.png')
