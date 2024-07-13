import matplotlib.image as mpimg
import math


def convert_gray_by_lightness(red, green, blue):

    # Initialize wide and height of image
    width = len(red[:, 0])
    height = len(red[0, :])

    result_image = [[0 for _ in range(height)] for _ in range(width)]

    for i in range(width):
        for j in range(height):
            result_image[i][j] = (
                (max(red[i, j], green[i, j], blue[i, j])) +
                (min(red[i, j], green[i, j], blue[i, j]))) / 2

    return tuple(result_image)


def convert_gray_by_avarage(red, green, blue):

    # Initialize wide and height of image
    width = len(red[:, 0])
    height = len(red[0, :])

    result_image = [[0 for _ in range(height)] for _ in range(width)]

    for i in range(width):
        for j in range(height):
            result_image[i][j] = sum([red[i, j], green[i, j], blue[i, j]]) / 3

    return tuple(result_image)


def convert_gray_by_luminosity(red, green, blue):

    # Initialize wide and height of image
    width = len(red[:, 0])
    height = len(red[0, :])

    result_image = [[0 for _ in range(height)] for _ in range(width)]

    for i in range(width):
        for j in range(height):
            result_image[i][j] = red[i, j] * 0.21 + \
                green[i, j] * 0.72 + blue[i, j] * 0.07

    return tuple(result_image)


if __name__ == "__main__":
    img = mpimg.imread("D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_1/dog.jpg")
    r_color = img[:, :, 0]
    g_color = img[:, :, 1]
    b_color = img[:, :, 2]

    # Exercise 12: Use Lightness to convert into gray image
    gray_image = convert_gray_by_lightness(r_color, g_color, b_color)
    print(gray_image[0][0])
    # -> Answer: A

    # Exercise 13: Use Average to convert into gray image
    gray_image = convert_gray_by_avarage(r_color, g_color, b_color)
    print(gray_image[0][0])
    # -> Answer: A

    # Exercise 14: Use Luminosity to convert into gray image
    gray_image = convert_gray_by_luminosity(r_color, g_color, b_color)
    print(gray_image[0][0])
    # -> Answer: C
