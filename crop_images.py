import argparse
import os
import cv2


def crop_and_save_images(directory, outputdir):
    """
    Crop the same region from multiple images in the given directory and save the cropped images,
    overwriting the original images.

    Args:
    directory: Path to the directory containing the images.
    outputdir: Path to the directory containing the output images.
    """
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

    # Read the first image to determine the region to crop
    first_image_path = os.path.join(directory, image_files[0])
    first_image = cv2.imread(first_image_path)

    # Define the region of interest (ROI) to crop (example: top-left corner)
    roi = (450, 350, 500, 400)  # Format: (start_x, start_y, width, height)

    # Crop and save each image
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        print("processing " + image_path)

        # Crop the region of interest
        cropped_image = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # Overwrite the original image with the cropped region
        cv2.imwrite(os.path.join(outputdir,image_file), cropped_image)

# Example usage:
# Specify the directory containing the images
# directory = "/path/to/your/image/directory"
# crop_and_save_images(directory)

def main():
    directory = "22"
    outputdir = "22"
    crop_and_save_images(directory, outputdir)

if __name__ == "__main__":
    main()