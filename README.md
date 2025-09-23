# CV-1-26: Circle Detection and Analysis

## Overview

This project solves problem **CV-1-26**. The program processes synthetic images containing a single perfect circle on a contrasting background. It generates a binary mask of the image, calculates the circle's area in pixels, and estimates its roundness.

---

## Features

- Reads an image of a circle (any location, size, and color contrast against the background).  
- Generates a **binary mask** where the circle is white and the background is black.  
- Calculates the **roundness** of detected objects using the formula: 4𝝅 * S(object) / P(object) ^ 2
- Determines which object is the actual circle based on the highest circularity value.  
- Computes and prints the **area of the circle in pixels**.  
- Saves the binary mask as a PNG file: <original_filename>_bin_mask.png
- 
- ## Usage
1. Place the image file in the same directory as the program, or provide the full path.  
2. Run the program:  

```bash
python3 main.py
```
3. Enter the filename (with extension) when prompted.
4. The program outputs the area of the circle in pixels and saves the binary mask in the same directory.
## Example
Suppose you have an image circle.png in the project directory:
```bash
python3 main.py
Enter the file name with extension: red.png
Object area 29171 pixels
The binary mask red_bin_mask.png will be saved in the same directory.
```
## Dependencies
The program requires the following Python packages:
1. opencv-python
2. numpy
3. matplotlib

You can enable dependencies by going to the project directory in the terminal and entering:
```bash
source venv/bin/activate
```
## Notes
1. Only a single circle per image is supported.
2. The circle can be of any size or color, as long as it contrasts with the background.
3. Roundness is used to ensure the circle is correctly identified, even if the mask inversion occurs.
