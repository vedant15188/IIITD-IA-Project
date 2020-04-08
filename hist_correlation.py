from PIL import Image
import numpy as np
import cv2
import math
import operator

'''
def make_hist(image, width, height):
    hist = [0] * 16777216
    for row in range(width):
        for col in range(height):
            r = image[row, col][0]
            g = image[row, col][1]
            b = image[row, col][2]
            hist[int_value] += 1
    return hist
'''

def make_hist(image, width, height):
    hist = [0] * 256
    for row in range(width):
        for col in range(height):
            pixel = image[row, col]
            hist[pixel] += 1
    return hist

def correlation(hist1, hist2):
    mean1 = sum(hist1) / float(len(hist1))
    mean2 = sum(hist2) / float(len(hist2))

    numerator = 0.0

    for i in range(len(hist1)):
        numerator += float(((hist1[i] - mean1) * (hist2[i] - mean2)))

    denominator = 0.0

    for i in range(len(hist1)):
        X = np.sum((hist1[i] - mean1) ** 2)
        Y = np.sum((hist2[i] - mean2) ** 2)
        denominator += float((X * Y) ** 0.5)

    correlation = numerator / denominator

    return correlation

def calculate(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('L')
    image2 = Image.open(image_path2).convert('L')

    image1_width, image1_height = image1.size
    image2_width, image2_height = image2.size

    image1_pixels = image1.load()
    image2_pixels = image2.load()

    hist1 = make_hist(image1_pixels, image1_width, image1_height)
    hist2 = make_hist(image2_pixels, image2_width, image2_height)

    return correlation(hist1, hist2)

    '''
    image1_r_histogram = make_hist(image1_pixels, image1_width, image1_height, 0) # R channel
    image1_g_histogram = make_hist(image1_pixels, image1_width, image1_height, 1) # G channel
    image1_b_histogram = make_hist(image1_pixels, image1_width, image1_height, 2) # B channel

    image2_r_histogram = make_hist(image2_pixels, image2_width, image2_height, 0) # R channel
    image2_g_histogram = make_hist(image2_pixels, image2_width, image2_height, 1) # R channel
    image2_b_histogram = make_hist(image2_pixels, image2_width, image2_height, 2) # R channel
    '''

    # im1 = cv2.imread(image_path1, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR
    # im2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

    # height1, width1, channels1 = im1.shape
    # height2, width2, channels2 = im2.shape



    # print(correlation(hist1, hist2))
    # print(chi_square(hist1, hist2))
    # print(bhattacharyya_distance(hist1, hist2))

    # hist1 = cv2.calcHist(hist1,[0], None, [16777216], [0,16777216])
    # hist2 = cv2.calcHist(hist2,[0], None, [16777216], [0,16777216])

    # a = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # print(a)

    # r_correlation = correlation(image1_r_histogram, image2_r_histogram)
    # g_correlation = correlation(image1_g_histogram, image2_g_histogram)
    # b_correlation = correlation(image1_b_histogram, image2_b_histogram)

    # print(r_correlation)
    # print(g_correlation)
    # print(b_correlation)

    '''
    image1_histogram = [0] * 256
    image2_histogram = [0] * 256

    mean1 = sum(image1_histogram) / float(len(image1_histogram))
    mean2 = sum(image2_histogram) / float(len(image2_histogram))

    correlation_numerator = 0.0

    for i in range(len(image1_histogram)):
        correlation_numerator += float(((image1_histogram[i] - mean1) * (image2_histogram[i] - mean2)))

    correlation_denominator = 0.0

    for i in range(len(image1_histogram)):
        X = np.sum((image1_histogram[i] - mean1) ** 2)
        Y = np.sum((image2_histogram[i] - mean2) ** 2)
        correlation_denominator += float((X * Y) ** 0.5)

    correlation = correlation_numerator / correlation_denominator

    print(correlation)
    '''

'''
100_Euro.jpg
100_Rupee.jpg
2000_Rupee.jpg
'''
