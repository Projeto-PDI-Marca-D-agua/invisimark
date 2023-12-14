import numpy as np
import cv2


class LSBImage:
    @staticmethod
    def embed_LSB_image(original_image, watermark_image):

        cover_image = original_image
        watermark = watermark_image

        if watermark.shape[0] > cover_image.shape[0] or watermark.shape[1] > cover_image.shape[1]:
            watermark = cv2.resize(
                watermark, (cover_image.shape[1], cover_image.shape[0]))

        for y in range(watermark.shape[0]):
            for x in range(watermark.shape[1]):
                for c in range(3):
                    cover_pixel_value = cover_image[y, x, c]

                    watermark_pixel_value = watermark[y, x, c]

                    modified_cover_pixel_value = (cover_pixel_value & 254) | (
                        (watermark_pixel_value >> 7) & 1)

                    cover_image[y, x, c] = modified_cover_pixel_value

        return cover_image

    @staticmethod
    def blind_extraction_LSB(marked_image):

        marked_img = marked_image

        revealed_watermark = np.zeros_like(marked_img)

        for y in range(marked_img.shape[0]):
            for x in range(marked_img.shape[1]):
                for c in range(3):
                    pixel_value = marked_img[y, x, c]

                    revealed_pixel_value = (pixel_value & 1) * 255

                    revealed_watermark[y, x, c] = revealed_pixel_value

        return revealed_watermark
