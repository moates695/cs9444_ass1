import matplotlib.pyplot as plt

fig, axs = plt.subplots(18, 3)
for i in range(18):
    for j in range(3):
        axs[i, j].imshow(f'/plot/hid_full3_18_{j + 1}_{i}.png')
        axs[i, j].settitle(f'Hidden layer {j + 1} Node {i} ')