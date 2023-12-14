import numpy as np
import cv2


class LSBImage:
    @staticmethod
    def embed_LSB_image(original_image, watermark_image):

        cover_image = original_image
        watermark = watermark_image

        # Check if the dimensions of the watermark image are compatible with the cover image.
        # If not, resize the watermark image.
        if watermark.shape[0] > cover_image.shape[0] or watermark.shape[1] > cover_image.shape[1]:
            watermark = cv2.resize(
                watermark, (cover_image.shape[1], cover_image.shape[0]))

        # Iterate through each pixel of the watermark image
        for y in range(watermark.shape[0]):  # For each row
            for x in range(watermark.shape[1]):  # For each column
                # For each color channel (RGB)
                for c in range(3):
                    # Get the pixel value in the cover image
                    cover_pixel_value = cover_image[y, x, c]

                    # Get the pixel value in the watermark image
                    watermark_pixel_value = watermark[y, x, c]

                    # Hide the bits of the watermark image in the least significant bits of the cover image pixel
                    modified_cover_pixel_value = (cover_pixel_value & 254) | (
                        (watermark_pixel_value >> 7) & 1)

                    # Update the pixel value in the cover image
                    cover_image[y, x, c] = modified_cover_pixel_value

        return cover_image

    @staticmethod
    def blind_extraction_LSB(marked_image):

        marked_img = marked_image

        # Empty image for the revealed watermark
        revealed_watermark = np.zeros_like(marked_img)

        # Iterate through each pixel of the image with the hidden watermark
        for y in range(marked_img.shape[0]):  # For each row
            for x in range(marked_img.shape[1]):  # For each column
                # For each color channel (RGB)
                for c in range(3):
                    # Get the pixel value in the image with the hidden watermark
                    pixel_value = marked_img[y, x, c]

                    # Extract the bits of the watermark from the least significant bits of the pixel
                    revealed_pixel_value = (pixel_value & 1) * 255

                    # Update the pixel value in the revealed watermark image
                    revealed_watermark[y, x, c] = revealed_pixel_value

        return revealed_watermark
