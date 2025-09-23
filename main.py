import cv2
import numpy
import math
import matplotlib.pyplot
import os


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
        raise ValueError("No contours found in binary mask.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        raise ValueError("Contour has zero perimeter.")
    
    circularity = 4 * math.pi * area / (perimeter ** 2)
    return circularity


def main():
    """
    Loads an image, converts it to a binary mask, determines which version of the mask (original or inverted)
    contains a more circular object, visualizes the selected mask, saves it as an image, and calculates the area
    of the object in pixels.

    Steps performed:
    1. Prompts the user to input the filename of an image.
    2. Loads the image using OpenCV.
    3. If the image is not found, the program exits with an error message.
    4. Converts the image to grayscale and then binarizes it using Otsu's thresholding method.
    5. Calculates the circularity of the object in both the original and inverted masks.
    6. Selects the mask where the object is more circular (closer to a perfect circle).
    7. Visualizes and saves the selected binary mask as a PNG file.
    8. Calculates and prints the area of the white object (number of non-zero pixels) in the binary mask.

    Output:
    - Saves the binary mask image to disk.
    - Prints the object area (in pixels) to the console.
    """

    try:
        st = input("Enter the file name with extension: ")
        img = cv2.imread(st)

        if img is None:
            raise FileNotFoundError(f"File '{st}' not found.")

        # Split filename and extension
        filename, _ = os.path.splitext(st)

        # Grayscale conversion and binarization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Compute circularity
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

    except FileNotFoundError as e:
        print(f"ERROR: {e}")

    except ValueError as e:
        print(f"ERROR: {e}")

    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    main()