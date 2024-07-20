import cv2
import numpy as np


def compute_difference(bg_img, input_img):
    difference_single_channel = np.subtract(bg_img, input_img)
    return difference_single_channel


def compute_binary_mask(difference_single_channel):
    difference_binary = np.where(
        difference_single_channel > 0, 255, 0)
    return difference_binary


def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(
        bg1_image,
        ob_image
    )
    binary_mask = (difference_single_channel)
    output = np.where(binary_mask == 255, ob_image, bg2_image)

    return output


if __name__ == '__main__':
    bg1_image = cv2.imread(
        'D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_2/GreenBackground.png', 1)
    bg1_image = cv2.resize(bg1_image, (678, 381))

    ob_image = cv2.imread(
        'D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_2/Object.png', 1)
    ob_image = cv2.resize(ob_image, (678, 381))

    bg2_image = cv2.imread(
        'D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_2/NewBackground.jpg', 1)
    bg2_image = cv2.resize(bg2_image, (678, 381))

    output = replace_background(bg1_image, bg2_image, ob_image)

    cv2.imshow("image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
