import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m

def load_image(file_path):
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def create_mean_kernel(k):
    kernel = (1/k**2) * np.ones((k,k))
    return kernel


def apply_mean_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    mean_image = image.copy()
    for _ in range(n):
        mean_image = convolve_mean(mean_image, kernel, c)
    if padding:
        mean_image = mean_image[c:-c, c:-c]
    return mean_image

def create_median_kernel(k):
    kernel_median = np.ones((k,k))
    return kernel_median

def apply_median_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    median_image = image.copy()
    for _ in range(n):
        median_image = convolve_median(median_image, kernel, c)
    if padding:
        median_image = median_image[c:-c, c:-c]
    return median_image

def create_gaussian_kernel(k):
    x = np.arange(0, k)
    y = np.arange(0, k)
    x, y = np.meshgrid(x - k/2, y - k/2)
    sigma = 4 # std deviation
    a = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return a / a.sum()

def apply_gaussian_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    gaussian_image = image.copy()
    for _ in range(n):
        gaussian_image = convolve_gaussian(gaussian_image, kernel, c)
    if padding:
        gaussian_image = gaussian_image[c:-c, c:-c]
    return gaussian_image

def pad_image(image, c):
    padded_image = np.pad(image, ((c, c), (c, c)), mode='constant')
    return padded_image

def convolve_mean(image, kernel, c):
    new_image = image.copy()
    for x in range(c, image.shape[0]-c):
        for y in range(c, image.shape[1]-c):
            sub_image = image[x - c : x + c + 1, y - c : y + c + 1]
            mean = (sub_image * kernel).sum()
            new_image[x, y] = round(mean)
    return new_image

def convolve_median(image, kernel, c):
    new_image = image.copy()
    for x in range(c, image.shape[0]-c):
        for y in range(c, image.shape[1]-c):
            sub_image = image[x - c : x + c + 1, y - c : y + c + 1]
            median = np.median(sub_image * kernel)
            new_image[x, y] = m.ceil(median)
    return new_image

def convolve_gaussian(image, kernel, c):
    new_image = image.copy()
    for x in range(c, image.shape[0]-c):
        for y in range(c, image.shape[1]-c):
            sub_image = image[x - c : x + c + 1, y - c : y + c + 1]
            gaussian = (sub_image * kernel).sum()
            new_image[x, y] = round(gaussian)
    return new_image

def main():
    file_path = r'.\images\lena.png'
    img = load_image(file_path)
    k_mean = int(input('Informe a dimensão do kernel:'))
    k_gaussian = k_median = k_mean
    kernel_mean = create_mean_kernel(k_mean)
    kernel_median = create_median_kernel(k_median)
    kernel_gaussian = create_gaussian_kernel(k_gaussian)
    n = int(input('Informe o número de aplicações do filtro: '))
    padding = int(input('Você deseja realizar Padding? (1 para sim, 0 para não): '))
    mean_image = apply_mean_filter(img, kernel_mean, n, padding)
    median_image = apply_median_filter(img, kernel_median, n, padding)
    gaussian_image = apply_gaussian_filter(img, kernel_gaussian, n, padding)
    
    
    #padding = int(input('Padding para mediana? (1 for yes, 0 for no): ')

    plt.figure(figsize=(20,5))

    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(1, 4, 2)
    plt.imshow(mean_image, cmap='gray')
    plt.title('Filtro de Média')

    plt.subplot(1, 4, 3)
    plt.imshow(median_image, cmap='gray')
    plt.title('Filtro de Mediana')

    plt.subplot(1, 4, 4)
    plt.imshow(gaussian_image, cmap='gray')
    plt.title('Filtro Gaussiano')

    plt.show()

if __name__ == "__main__":
    main()