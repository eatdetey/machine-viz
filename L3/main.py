import cv2 as cv
import numpy as np
import random

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

kernel_3 = create_gaussian_kernel(3, 1)
kernel_5 = create_gaussian_kernel(5, 5)
kernel_7 = create_gaussian_kernel(7, 20)
# print(f"РАЗМЕР 3x3:\n{kernel_3}, \nsum:{np.sum(kernel_3)}")
# print(f"РАЗМЕР 5x5:\n{kernel_5}, \nsum:{np.sum(kernel_5)}")
# print(f"РАЗМЕР 7x7:\n{kernel_7}, \nsum:{np.sum(kernel_7)}")

# frame = cv.cvtColor(cv.imread("CHICKENJOKEY.jpg"), cv.COLOR_BGR2GRAY)
frame = cv.imread("CHICKENJOKEY.jpg")

frame_noise = np.copy(frame)
prob = 0.05
h, w = frame_noise.shape[:2]
num_pixels = int(h * w * prob)

for _ in range(num_pixels // 2):
    row = random.randint(0, h - 1)
    col = random.randint(0, w - 1)
    frame_noise[row, col] = [255, 255, 255]

for _ in range(num_pixels // 2):
    row = random.randint(0, h - 1)
    col = random.randint(0, w - 1)
    frame_noise[row, col] = [0, 0, 0]

frame_noise = cv.cvtColor(frame_noise, cv.COLOR_BGR2GRAY)

blur_frame_noise = apply_gaussian_filter(frame_noise, kernel_3)

# print(frame_noise[15,15])

colored_frame_noise = np.copy(frame)
colored_frame_noise = cv.cvtColor(colored_frame_noise, cv.COLOR_BGR2HSV)

# print(colored_frame_noise[15,15][2])

for i in range(colored_frame_noise.shape[0]):
    for j in range(colored_frame_noise.shape[1]):
        colored_frame_noise[i,j][2] = blur_frame_noise[i,j]

colored_frame_noise = cv.cvtColor(colored_frame_noise, cv.COLOR_HSV2BGR)

cv.imshow("Color image", frame)
cv.imshow("Noise GS image", frame_noise)
cv.imshow("Blurred (3x3)", blur_frame_noise)
cv.imshow("asfafs", colored_frame_noise)
# cv.imshow("Blurred (3x3)", apply_gaussian_filter(frame, kernel_3))
# cv.imshow("Blurred (7x7)", apply_gaussian_filter(frame, kernel_7))
# cv.imshow("Blurred OpenCV (3x3)", cv.GaussianBlur(frame, (3,3), 1))
# cv.imshow("Blurred OpenCV (7x7)", cv.GaussianBlur(frame, (7,7), 20))
cv.waitKey(0)
cv.destroyAllWindows()