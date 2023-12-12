import numpy as np
import cv2


class DCTText:
    @staticmethod
    def rgb_insert_text_dct(original_image, text):
        alpha = 0.01
        height, width, _ = original_image.shape

        text_array = np.array([ord(char) for char in text])

        blue_channel, green_channel, red_channel = cv2.split(original_image)

        blue_dct = cv2.dct(np.float32(blue_channel))
        green_dct = cv2.dct(np.float32(green_channel))
        red_dct = cv2.dct(np.float32(red_channel))

        for i in range(min(len(text_array), original_image.shape[0] * original_image.shape[1])):
            blue_dct[i // original_image.shape[1], i %
                     original_image.shape[1]] += alpha * text_array[i]

        blue_inverse = cv2.idct(blue_dct)
        green_inverse = cv2.idct(green_dct)
        red_inverse = cv2.idct(red_dct)

        marked_image = cv2.merge((blue_inverse, green_inverse, red_inverse))

        return marked_image

    @staticmethod
    def rgb_extract_text_dct(original_image, marked_image, text_length):
        alpha = 0.01
        blue_channel, green_channel, red_channel = cv2.split(marked_image)
        original_blue_channel, original_green_channel, original_red_channel = cv2.split(
            original_image)

        blue_marked_dct = cv2.dct(np.float32(blue_channel))

        blue_original_dct = cv2.dct(np.float32(original_blue_channel))

        extracted_text = []

        for i in range(min(text_length, blue_marked_dct.size)):
            diff_coefficient = blue_marked_dct[i // blue_marked_dct.shape[1], i % blue_marked_dct.shape[1]
                                               ] - blue_original_dct[i // blue_original_dct.shape[1], i % blue_original_dct.shape[1]]
            extracted_text.append(diff_coefficient / alpha)

        extracted_text = ''.join([chr(int(round(value)) % 256)
                                 for value in extracted_text])

        return extracted_text
