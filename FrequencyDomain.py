import matplotlib as ml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.fftpack as sfft

file_path = "shell.png"
image = np.asarray(Image.open(file_path).convert("L"))
freq = np.fft.fft2(image)
shiftedfreq = np.abs(freq)

# Z_fft = sfft.fft2(image)
# Z_shift = sfft.fftshift(Z_fft)
# shiftedfreq = np.abs(Z_shift)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
ax[0,0].hist(freq.ravel(), bins=100)
ax[0,0].set_title('hist(freq)')
ax[0,1].hist(np.log(shiftedfreq).ravel(), bins=100)
ax[0,1].set_title('hist(log(freq))')
ax[1,0].imshow(np.log(shiftedfreq), interpolation="none", cmap="gray")
ax[1,0].set_title('log(freq)')
ax[1,1].imshow(image, interpolation="none", cmap="gray")
plt.show()