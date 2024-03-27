import numpy as np

def mean_filter(image, kernel_size, n_applications= 1, padding= 1):
    # veirfying if the image is in grayscale or not
    if len(image.shape) == 2:
        row, columns = image.shape
    elif len(image.shape) == 3:
        row = image.shape[0]
        columns = image.shape[1]

    new_image = image.copy()
    kernel = np.ones((kernel_size, kernel_size))*1/(kernel_size**2)
    center = int(kernel_size/2)

    if padding == 1:
        new_pixels = 2 * center # Necessary new pixels for padding
        pdd = np.zeros((int(row + new_pixels), int(columns + new_pixels)))
        new_rows, new_columns = pdd.shape
        # pdd[center: n_rows - center, center: n_columns - center] == image
        new_image = pdd.copy()

    for i in range(n_applications):
        for x_i in range(center, new_rows - center): # original image initial horizontal pixel
            for y_i in range(center, new_columns - center): # original image initial vertical pixel
                aux = new_image[x_i - center: x_i + center + 1, y_i - center: y_i + center + 1]
                mean = (aux * kernel).sum()
                new_image[x_i, y_i] = round(mean)
                   
    if padding == 1:
        final_image = np.zeros(image.shape)
        final_image = new_image[center: new_rows - center, center: new_columns - center]
    else:
        final_image = np.zeros(image.shape)
        final_image = new_image
    return new_image