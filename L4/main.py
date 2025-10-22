import cv2 as cv
import numpy as np

def create_gaussian_kernel(size, sigma):
    center = size // 2
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)

    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    return kernel

def apply_gaussian_filter(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='reflect')

    result = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.sum(region * kernel)

    return result.astype(np.uint8)

def get_quantized_angle(grad_x, grad_y, tg):
    quantized_angle = np.zeros_like(grad_x, dtype=np.uint8)

    mask0 = ((grad_x > 0) & (grad_y < 0) & (tg < -2.414)) | ((grad_x < 0) & (grad_y < 0) & (tg > 2.414))
    quantized_angle[mask0] = 0

    mask1 = (grad_x > 0) & (grad_y < 0) & (tg >= -2.414) & (tg < -0.414)
    quantized_angle[mask1] = 1

    mask2 = ((grad_x > 0) & (grad_y < 0) & (tg >= -0.414)) | ((grad_x > 0) & (grad_y > 0) & (tg < 0.414))
    quantized_angle[mask2] = 2

    mask3 = (grad_x > 0) & (grad_y > 0) & (tg >= 0.414) & (tg < 2.414)
    quantized_angle[mask3] = 3

    mask4 = ((grad_x > 0) & (grad_y > 0) & (tg >= 2.414)) | ((grad_x < 0) & (grad_y > 0) & (tg <= -2.414))
    quantized_angle[mask4] = 4

    mask5 = (grad_x < 0) & (grad_y > 0) & (tg > -2.414) & (tg <= -0.414)
    quantized_angle[mask5] = 5

    mask6 = ((grad_x < 0) & (grad_y > 0) & (tg > -0.414)) | ((grad_x < 0) & (grad_y < 0) & (tg < 0.414))
    quantized_angle[mask6] = 6

    mask7 = (grad_x < 0) & (grad_y < 0) & (tg >= 0.414) & (tg < 2.414)
    quantized_angle[mask7] = 7

    return quantized_angle

def canny_edge_detector(path):
    img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel = create_gaussian_kernel(7, 20)
    blur_img = apply_gaussian_filter(gray_img, kernel)
    gray_img = blur_img.astype(np.float64)

    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    grad_x = np.zeros_like(gray_img)
    grad_y = np.zeros_like(gray_img)

    padded = np.pad(gray_img, ((1, 1), (1, 1)), mode='reflect')

    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            region = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_kernel_x)
            grad_y[i, j] = np.sum(region * sobel_kernel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_x_safe = np.where(grad_x == 0, 1e-6, grad_x)
    tg = grad_y / grad_x_safe

    quantized_angle = get_quantized_angle(grad_x, grad_y, tg)

    nms = np.zeros_like(magnitude, dtype=np.uint8)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = quantized_angle[i, j]
            mag = magnitude[i, j]

            if direction in [0, 4]:
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif direction in [1, 5]:
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            elif direction in [2, 6]:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            else:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

            if mag > max(neighbors):
                nms[i, j] = 255
            else:
                nms[i, j] = 0

    max_grad = np.max(magnitude)
    low_level = max_grad // 5
    high_level = max_grad // 3

    strong_edges = (magnitude >= high_level)
    weak_edges = ((magnitude >= low_level) & (magnitude < high_level))

    result = np.zeros_like(nms, dtype=np.uint8)

    result[strong_edges & (nms == 255)] = 255

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] and nms[i, j] == 255:
                region = result[i - 1:i + 2, j - 1:j + 2]
                if np.any(region == 255):
                    result[i, j] = 255

    overlay = img.copy()
    overlay[result == 255] = [0, 0, 255]

    cv.imshow("Grayscaled and blurred", blur_img)
    cv.imshow("Non-Maximum Suppression", nms)
    cv.imshow("Double Threshold Result", result)
    cv.imshow("Contoured Image", overlay)

    print(f"Magnitude: \n{magnitude}")
    print(f"Angle: \n{quantized_angle}")

canny_edge_detector('gatito.jpeg')
cv.waitKey(0)
cv.destroyAllWindows()

