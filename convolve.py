import cv2 
import numpy as np
import time
import matplotlib as ml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.fftpack as sfft
import tensorflow as tf

def processImage(image): 
  image = cv2.imread(image) 
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
  return image

  
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

if __name__ == '__main__':
    # Grayscale Image
    file_path = "shell.png"
    image = processImage(file_path)
    cv2.imwrite('grayscale.png', image)

    # Conv2D Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    start = time.time()
    output = convolve2D(image, kernel, padding=2)
    print("2D Convolution Time:", time.time() - start, "ms")
    cv2.imwrite('2DConvolved.jpg', output)

    # Do the same thing with FFT
    im = np.asarray(Image.open(file_path).convert("L"))
    fft_length1 = tf.shape(im)[0]
    fft_length2 = tf.shape(im)[1]
    start = time.time()
    im_fft = tf.signal.rfft2d(im, fft_length=[fft_length1, fft_length2])
    kernel_fft = tf.signal.rfft2d(kernel, fft_length=[fft_length1, fft_length2])
    im_convolved = np.array(tf.signal.irfft2d(im_fft * kernel_fft, [fft_length1, fft_length2]))
    print("FFT Convolution Time:", time.time() - start, "ms")
    cv2.imwrite('FFTConvolved.jpg', im_convolved)