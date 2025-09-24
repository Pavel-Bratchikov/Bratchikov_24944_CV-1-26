import cv2
import numpy
import math
import matplotlib.pyplot
import os


def circular_object_circularity(binary_mask):
    """
    Estimate the circularity of the largest white object in a binary mask.

    Circularity is calculated as:

        C = 4 * pi * (Area / Perimeter^2)

    Parameters
    ----------
    binary_mask : numpy.ndarray
        A binary image (0 and 255 values).

    Returns
    -------
    float
        The circularity value of the largest contour.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in binary mask.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        raise ValueError("Contour has zero perimeter.")
    
    return 4 * math.pi * area / (perimeter ** 2)


def binarize_image(img):
    """
    Convert an image to a binary mask using grayscale and Otsu's thresholding.

    Parameters
    ----------
    img : numpy.ndarray
        Original image.

    Returns
    -------
    numpy.ndarray
        Binary mask (0 and 255 values).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def select_circular_mask(mask):
    """
    Compare circularity of original and inverted masks.
    Select the mask with the more circular object.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.

    Returns
    -------
    numpy.ndarray
        The mask with higher circularity.
    """
    circ1 = circular_object_circularity(mask)
    circ2 = circular_object_circularity(cv2.bitwise_not(mask))
    return mask if circ1 >= circ2 else cv2.bitwise_not(mask)


def main():
    """
    Main program logic:
    - Load image
    - Binarize
    - Choose the most circular mask
    - Save binary mask as PNG
    - Print object area
    """
    try:
        st = input("Enter the file name with extension: ")
        img = cv2.imread(st)

        if img is None:
            raise FileNotFoundError(f"File '{st}' not found.")

        # Split filename and extension
        filename, _ = os.path.splitext(st)

        # Binarization
        mask = binarize_image(img)

        # Select more circular mask
        mask = select_circular_mask(mask)

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