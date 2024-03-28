import numpy as np
import matplotlib.pyplot as plt
import cv2 

def histogram(image):
    x_pixels, y_pixels = image.shape
    hist = np.zeros(256)

    for x in range(x_pixels):
        for y in range(y_pixels):
            hist[image[x,y]] += 1

    return hist

def threshold(image, threshold):
    threshold_image = image.copy()
    threshold_image[image < threshold] = 0
    threshold_image[image >= threshold] = 255
    return threshold_image

def main():
  img = cv2.imread(r'..\images\lena.png')
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  y = histogram(img_gray)
  x = np.arange(256)

  t_image = threshold(img_gray, 200)

  plt.figure(figsize=(10,6))
  plt.title('Image Histogram')
  plt.bar(x, y, width= 0.3)

  plt.figure(figsize=(10,6))
  plt.title('Imagem com limiarização')
  plt.imshow(t_image, cmap='grey')
  plt.show()

if __name__ == '__main__':
    main()