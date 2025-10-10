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

kernel_3 = create_gaussian_kernel(3, 1)
kernel_5 = create_gaussian_kernel(5, 5)
kernel_7 = create_gaussian_kernel(7, 20)
# print(f"РАЗМЕР 3x3:\n{kernel_3}, \nsum:{np.sum(kernel_3)}")
# print(f"РАЗМЕР 5x5:\n{kernel_5}, \nsum:{np.sum(kernel_5)}")
# print(f"РАЗМЕР 7x7:\n{kernel_7}, \nsum:{np.sum(kernel_7)}")

frame = cv.cvtColor(cv.imread("CHICKENJOKEY.jpg"), cv.COLOR_BGR2GRAY)

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

cv.imshow("Usual horse", frame)
# cv.imshow("Blurred (3x3)", apply_gaussian_filter(frame, kernel_3))
# cv.imshow("Blurred (7x7)", apply_gaussian_filter(frame, kernel_7))
cv.imshow("Blurred OpenCV (3x3)", cv.GaussianBlur(frame, (3,3), 1))
cv.imshow("Blurred OpenCV (7x7)", cv.GaussianBlur(frame, (7,7), 20))
cv.waitKey(0)
cv.destroyAllWindows()