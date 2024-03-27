import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m

def load_image(file_path):
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def pad_image(image, c):
    padded_image = np.pad(image, ((c, c), (c, c)), mode='constant')
    return padded_image

def convolve(image, kernel, c):
    height, width = image.shape
    k_height, k_width = kernel.shape

    # Criando uma matriz de zeros para armazenar o resultado da convolução
    result = np.zeros_like(image, dtype=float)

    # Aplicando a convolução
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
    kernel_1 = np.array([[ 0, 1, 0],
                         [ 1,-4, 1],
                         [ 0, 1, 0]])
    kernel_2 = np.array([[ 1,  1,  1],
                         [ 1, -8,  1],
                         [ 1,  1,  1]])
    kernel_3 = np.array([[  0,-1,  0],
                         [ -1, 4, -1],
                         [ 0, -1,  0]])
    kernel_4 = np.array([[-1, -1, -1],
                         [ -1, 8, -1],
                         [ -1,-1, -1]])
    
    return kernel_1, kernel_2, kernel_3, kernel_4

def apply_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    filtered_image_x = image.copy()
    for _ in range(n):
        filtered_image_x = convolve(filtered_image_x, kernel, c)
    filtered_image_x[filtered_image_x < 0] = 0
    return filtered_image_x

def apply_x_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    filtered_image_x = image.copy()
    for _ in range(n):
        filtered_image_x = convolve(filtered_image_x, kernel, c)
    filtered_image_x[filtered_image_x < 0] = 0
    return filtered_image_x

def apply_y_filter(image, kernel, n, padding):
    c = int(kernel.shape[0] / 2)
    if padding:
        image = pad_image(image, c)
    filtered_image_y = image.copy()
    for _ in range(n):
        filtered_image_y = convolve(filtered_image_y, kernel, c)
    filtered_image_y[filtered_image_y < 0] = 0
    return filtered_image_y

def main():
    file_path = r'..\images\lena.png'
    img = load_image(file_path)
    kernel_s_x, kernel_s_y = create_sobel_kernel()
    kernel_p_x, kernel_p_y = create_prewitt_kernel()
    kernel_l_1, kernel_l_2, kernel_l_3, kernel_l_4 = create_laplacian_kernel()
    n = int(input('Informe o número de aplicações do filtro: '))
    padding = int(input('Você deseja realizar Padding? (1 para sim, 0 para não): '))
    sobel_x = apply_x_filter(img, kernel_s_x, n, padding)
    sobel_y = apply_y_filter(img, kernel_s_y, n, padding)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    prewitt_x = apply_x_filter(img, kernel_p_x, n, padding)
    prewitt_y = apply_y_filter(img, kernel_p_y, n, padding)
    prewitt = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian_1 = apply_filter(img, kernel_l_1, n, padding)
    laplacian_2 = apply_filter(img, kernel_l_2, n, padding)
    laplacian_3 = apply_filter(img, kernel_l_3, n, padding)    
    laplacian_4 = apply_filter(img, kernel_l_4, n, padding)

    # Normalizando os resultados para o intervalo [0, 1]
    sobel_x = sobel_x / np.max(sobel_x)
    sobel_y = sobel_y / np.max(sobel_y)

    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalizando os resultados para o intervalo [0, 1]
    prewitt_x = prewitt_x / np.max(prewitt_x)
    prewitt_y = prewitt_y / np.max(prewitt_y)

    prewitt = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalizando os resultados para o intervalo [0, 1]

    laplacian_1 = laplacian_1/ np.max(laplacian_1) 
    laplacian_2 = laplacian_2/ np.max(laplacian_2)
    laplacian_3 = laplacian_3/ np.max(laplacian_3)
    laplacian_4 = laplacian_4/ np.max(laplacian_4)

    plt.figure(figsize=(15,5))

    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(1, 5, 2)
    plt.imshow(laplacian_1, cmap='gray')
    plt.title('Filtro Laplaciano 1')

    plt.subplot(1, 5, 3)
    plt.imshow(laplacian_2, cmap='gray')
    plt.title('Filtro Laplaciano 2')
    
    plt.subplot(1, 5, 4)
    plt.imshow(laplacian_3, cmap='gray')
    plt.title('Filtro Laplaciano 3')

    plt.subplot(1, 5, 5)
    plt.imshow(laplacian_4, cmap='gray')
    plt.title('Filtro Laplaciano 4')


    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Filtro Prewitt Y')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sobel, cmap='gray')
    plt.title('Filtro Sobel')

    plt.subplot(1, 3, 3)
    plt.imshow(prewitt, cmap='gray')
    plt.title('Filtro Prewitt')
    
    plt.show()

if __name__ == "__main__":
    main()