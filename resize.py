from PIL import Image

def resize_image(input_image_path, output_image_path, size, keep_aspect_ratio=True):
    """
    Resizes an image to a specific size.

    Args:
        input_image_path (str): The path to the input image file.
        output_image_path (str): The path to save the resized image.
        size (tuple): A tuple containing the desired (width, height).
        keep_aspect_ratio (bool): If True, resizes the image while maintaining
                                  its original aspect ratio. If False, forces
                                  the image to the exact specified dimensions,
                                  which may result in stretching.
    """
    try:
        # Open the image file
        with Image.open(input_image_path) as img:
            if keep_aspect_ratio:
                # Calculate the new size while maintaining the aspect ratio
                img.thumbnail(size)
            else:
                # Resize to the exact specified dimensions
                img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save the resized image
            img.save(output_image_path)
            print(f"Image resized successfully and saved to: {output_image_path}")

    except IOError as e:
        print(f"Error: Unable to open or process the image file. {e}")

# --- Example usage ---

# Define file paths and desired size
input_file = 'ocr-l.png'
output_file_fixed = '2700x1800.png'
target_size = (2700, 1800) # (width, height)

# To resize and potentially stretch the image to the exact size
# Note: For this example, you need an image file named 'my_original_image.jpg' in the same directory.
resize_image(input_file, output_file_fixed, target_size, keep_aspect_ratio=False)

# # To resize and maintain the aspect ratio, fitting within the target size
# # Note: The output image may be smaller than the target_size in one dimension.
# resize_image(input_file, output_file_aspect_ratio, target_size, keep_aspect_ratio=True)
