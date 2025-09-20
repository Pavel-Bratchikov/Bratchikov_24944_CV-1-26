import cv2
import numpy
import math
import matplotlib.pyplot


def circular_object_circularity(binary_mask):
    """
    A function for estimating the roundness of a white object on a mask
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("NO CONTOURS!!")
        exit()
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        print("NO PERIMETER!!")
        exit()
    
    circularity = 4 * math.pi * area / (perimeter ** 2)
    return circularity

st = input("Enter the file name with extension: ")
img = cv2.imread(st)

if img is None:
    print("")
    exit()

# We are looking for a name without an extension
k = 0
for i in st[::-1]:
    k += 1
    if i == '.':
        break

# Grayscale conversion and binarization
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# We leave the version where the object is "rounder"
if circular_object_circularity(cv2.bitwise_not(mask)) > circular_object_circularity(mask):
    mask = cv2.bitwise_not(mask)

# Visualization
matplotlib.pyplot.imshow(mask, cmap='gray')
matplotlib.pyplot.axis('off')
matplotlib.pyplot.title("Бинарная маска")
matplotlib.pyplot.savefig(st[0:len(st) - k] + "_bin_mask.png", bbox_inches='tight', pad_inches=0)

# Calculating the area of ​​a white object
area = numpy.count_nonzero(mask) 
print(f"Object area {area} pixels")