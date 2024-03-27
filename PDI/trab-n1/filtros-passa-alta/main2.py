import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img

def pad_image(image, c):
    padded_image = np.pad(image, ((c, c), (c, c)), mode='constant')
    return padded_image

def convolve(image, kernel, c):
    height, width = image.shape
    k_height, k_width = kernel.shape
    result = np.zeros_like(image, dtype=float)

    for y in range(c, height - c):
        for x in range(c, width - c):
            region = image[y-c:y+c+1, x-c:x+c+1]
            result[y, x] = np.sum(region * kernel)

    return result

def create_sobel_kernel():
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    return kernel_x, kernel_y

def create_prewitt_kernel():
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])
    return kernel_x, kernel_y

def create_laplacian_kernel():
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])
    return kernel

def apply_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    filtered_image = image.copy()
    for _ in range(n):
        filtered_image = convolve(filtered_image, kernel, c)
    filtered_image[filtered_image < 0] = 0
    return filtered_image

def main():
    file_path = r'..\images\lena.png'
    img = load_image(file_path)
    n = int(input('Informe o número de aplicações do filtro: '))
    padding = int(input('Você deseja realizar Padding? (1 para sim, 0 para não): '))

    # Sobel
    kernel_s_x, kernel_s_y = create_sobel_kernel()
    sobel_x = apply_filter(img, kernel_s_x, n, padding)
    sobel_y = apply_filter(img, kernel_s_y, n, padding)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Prewitt
    kernel_p_x, kernel_p_y = create_prewitt_kernel()
    prewitt_x = apply_filter(img, kernel_p_x, n, padding)
    prewitt_y = apply_filter(img, kernel_p_y, n, padding)
    prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Laplaciano
    kernel_l = create_laplacian_kernel()
    laplacian = apply_filter(img, kernel_l, n, padding)

    # Plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(1, 4, 2)
    plt.imshow(sobel, cmap='gray')
    plt.title('Filtro Sobel')

    plt.subplot(1, 4, 3)
    plt.imshow(prewitt, cmap='gray')
    plt.title('Filtro Prewitt')

    plt.subplot(1, 4, 4)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Filtro Laplaciano')

    plt.show()

if __name__ == "__main__":
    main()