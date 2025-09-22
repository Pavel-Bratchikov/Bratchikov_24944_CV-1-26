import cv2
import numpy
import math
import matplotlib.pyplot
import os
import sys


def circular_object_circularity(binary_mask):
    """
    Estimate the circularity of the largest white object in a binary mask.

    Circularity is a measure of how close a shape is to a perfect circle,
    calculated as:

        C = 4 * pi * (Area / Perimeter^2)

    where:
        - Area is the contour area of the object
        - Perimeter is the contour perimeter

    For a perfect circle, circularity is approximately 1.0.
    For non-circular shapes, circularity decreases.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        A binary image (0 and 255 values) where the object is expected
        to be white and the background black.

    Returns
    -------
    float
        The circularity value of the largest contour.
        If no contour or perimeter is found, the program terminates.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("NO CONTOURS!!")
        sys.exit(1)
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        print("NO PERIMETER!!")
        sys.exit(1)
    
    circularity = 4 * math.pi * area / (perimeter ** 2)
    return circularity



st = input("Enter the file name with extension: ")
img = cv2.imread(st)

if img is None:
    print("File not found!")
    sys.exit(1)

# Split filename and extension
filename, _ = os.path.splitext(st)

# Grayscale conversion and binarization
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

circ1 = circular_object_circularity(mask)
circ2 = circular_object_circularity(cv2.bitwise_not(mask))

# We leave the version where the object is "rounder"
if circ2 > circ1:
    mask = cv2.bitwise_not(mask)

# Visualization
matplotlib.pyplot.imshow(mask, cmap='gray')
matplotlib.pyplot.axis('off')
matplotlib.pyplot.title("Binary Mask")
matplotlib.pyplot.savefig(filename + "_bin_mask.png", bbox_inches='tight', pad_inches=0)

# Calculating the area of a white object
area = numpy.count_nonzero(mask) 
print(f"Object area {area} pixels")