import numpy as np
import cv2


class DCTImage:
    @staticmethod
    def rgb_insert_dct(original_image, watermark):
        alpha = 0.01
        DCTImage.resize(original_image, watermark)
        blue_channel, green_channel, red_channel = cv2.split(original_image)

        blue_dct = cv2.dct(np.float32(blue_channel))
        green_dct = cv2.dct(np.float32(green_channel))
        red_dct = cv2.dct(np.float32(red_channel))

        blue_watermark = alpha * watermark[:, :, 0]
        green_watermark = alpha * watermark[:, :, 1]
        red_watermark = alpha * watermark[:, :, 2]

        blue_dct += blue_watermark
        green_dct += green_watermark
        red_dct += red_watermark

        blue_inverse = cv2.idct(blue_dct)
        green_inverse = cv2.idct(green_dct)
        red_inverse = cv2.idct(red_dct)

        marked_image = cv2.merge((blue_inverse, green_inverse, red_inverse))

        return marked_image

    @staticmethod
    def rgb_remove_dct(original_image, marked_image):
        alpha = 0.01

        blue_channel_M, green_channel_M, red_channel_M = cv2.split(
            marked_image)

        blue_channel_M = cv2.dct(np.float32(blue_channel_M))
        green_channel_M = cv2.dct(np.float32(green_channel_M))
        red_channel_M = cv2.dct(np.float32(red_channel_M))

        blue_channel_O, green_channel_O, red_channel_O = cv2.split(
            original_image)

        blue_channel_O = cv2.dct(np.float32(blue_channel_O))
        green_channel_O = cv2.dct(np.float32(green_channel_O))
        red_channel_O = cv2.dct(np.float32(red_channel_O))

        blue_channel_watermark = (blue_channel_M - blue_channel_O) / alpha
        green_channel_watermark = (green_channel_M - green_channel_O) / alpha
        red_channel_watermark = (red_channel_M - red_channel_O) / alpha

        extracted_watermark = cv2.merge(
            (blue_channel_watermark, green_channel_watermark, red_channel_watermark))

        return extracted_watermark

    @staticmethod
    def resize(original_image, watermark):
        return cv2.resize(watermark, (original_image.shape[1], original_image.shape[0]))
