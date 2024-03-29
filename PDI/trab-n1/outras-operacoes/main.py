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

def multithreshold(image, thresholds: list):
 
    output_image = np.zeros_like(image)
    for i in range(len(thresholds) + 1):
        if i == 0:
            output_image[image <= thresholds[i]] = 0
        elif i == len(thresholds):
            output_image[image > thresholds[i - 1]] = 255
        else:
            output_image[(image > thresholds[i - 1]) & (image <= thresholds[i])] = 127
    return output_image

def histogram_equalization(image):
    # Obter a contagem de pixels para cada valor de intensidade
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0,256))
    
    # Calculando a função de distribuição acumulada (CDF)
    cdf = hist.cumsum()
    
    # Normalizar a CDF para a faixa de intensidade da imagem
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Interpolação dos valores CDF normalizados
    cdf_interpolated = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    # Reformatar a imagem com base na CDF normalizada interpolada
    image_equalized = cdf_interpolated.reshape(image.shape)
    
    return image_equalized.astype(np.uint8)

def main():
    img = cv2.imread(r'..\images\lena.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y = histogram(img_gray)
    x = np.arange(256)

    t_image = threshold(img_gray, 200)
    multi_t_image = multithreshold(img_gray, [100, 200])
    equalized_img = histogram_equalization(img_gray)
    y_e = histogram(equalized_img)

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.title('Image Histogram')
    plt.bar(x, y, width= 0.3)
    plt.subplot(1,2,2)
    plt.title('Image Equalized Histogram')
    plt.bar(x, y_e, width= 0.3)

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.title('Image Original')
    plt.imshow(img_gray, cmap='grey')
    plt.subplot(1,2,2)
    plt.title('Image Equalized')
    plt.imshow(equalized_img, cmap='grey')


    plt.figure(figsize=(10,6))
    plt.title('Imagem com limiarização')
    plt.imshow(t_image, cmap='grey')

    plt.figure(figsize=(10,6))
    plt.title('Imagem com multilimiarização')
    plt.imshow(multi_t_image, cmap='grey')
    plt.show()

if __name__ == '__main__':
    main()