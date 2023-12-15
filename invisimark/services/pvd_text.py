import cv2
import numpy as np
import os
import math


class PVDText:

    @staticmethod
    def hide_message(image, text):
        text_bytes = text.encode('utf-8')

        new_image = PVDText.compare_area_string(image, text_bytes)

        stego_image_r, secret_message_r = PVDText.hide_message_rgb(
            new_image, text_bytes, 'r')
        stego_image_g, secret_message_g = PVDText.hide_message_rgb(
            new_image, text_bytes, 'g')
        stego_image_b, secret_message_b = PVDText.hide_message_rgb(
            new_image, text_bytes, 'b')

        psnr_r = PVDText.psnr(image, stego_image_r)
        psnr_g = PVDText.psnr(image, stego_image_g)
        psnr_b = PVDText.psnr(image, stego_image_b)

        best_channel = np.argmax([psnr_r, psnr_g, psnr_b])

        stego_image = stego_image_r if best_channel == 0 else stego_image_g if best_channel == 1 else stego_image_b
        secret_message = secret_message_r if best_channel == 0 else secret_message_g if best_channel == 1 else secret_message_b
        print(secret_message)

        return stego_image, secret_message

    @staticmethod
    def hide_message_rgb(cover_image, message, rgb):
        stego_image = np.copy(cover_image)

        cover_image = PVDText.compare_area_string(cover_image, message)

        cover_b, cover_g, cover_r = cv2.split(cover_image)

        if rgb == 'r':
            cover_channel = cover_r
        elif rgb == 'g':
            cover_channel = cover_g
        elif rgb == 'b':
            cover_channel = cover_b
        else:
            raise ValueError("rgb must be 'r', 'g', or 'b'")

        for i in range(len(message)):
            message_bit = int(message[i])

            y = i // cover_channel.shape[1]
            x = i % cover_channel.shape[1]
            pixel = cover_channel[y, x]

            pixel &= 0xFE

            pixel |= message_bit

            cover_channel[y, x] = pixel

        if rgb == 'r':
            stego_image = cv2.merge((cover_b, cover_g, cover_channel))
        elif rgb == 'g':
            stego_image = cv2.merge((cover_b, cover_channel, cover_r))
        elif rgb == 'b':
            stego_image = cv2.merge((cover_channel, cover_g, cover_r))

        return stego_image, message

    @staticmethod
    def extract_message(stego_image):
        stego_b, stego_g, stego_r = cv2.split(stego_image)

        message_bits = {'r': [], 'g': [], 'b': []}

        for channel, stego_channel in zip(['r', 'g', 'b'], [stego_r, stego_g, stego_b]):
            for y in range(stego_channel.shape[0]):
                for x in range(stego_channel.shape[1]):
                    bit_message = stego_channel[y, x] & 0x01

                    message_bits[channel].append(bit_message)

        message = {channel: ''.join(chr(int(''.join(str(bit) for bit in message_bits[channel][i:i+8]), 2)) for i in range(
            0, len(message_bits[channel]) - len(message_bits[channel]) % 8, 8)) for channel in ['r', 'g', 'b']}

        return message

    @staticmethod
    def compare_area_string(image, message):
        image_area = image.shape[0] * image.shape[1]

        if image_area >= len(message):
            new_image = image
        else:
            multiplier = len(message) / image_area
            new_height = math.sqrt(multiplier) * image.shape[0]
            new_width = math.sqrt(multiplier) * image.shape[1]
            new_image = cv2.resize(image, (new_width, new_height))

        return new_image

    @staticmethod
    def compare_messages(decoded_message):
        file_path = os.path.join('watermarked_images', 'message.txt')

        with open(file_path, 'r') as file:
            saved_strings = file.read().splitlines()

        comparison_result = {}

        decoded_string = {}

        for channel in ['r', 'g', 'b']:
            try:
                decoded_string[channel] = bytes(
                    decoded_message[channel], 'latin1').decode('utf-8')
            except UnicodeDecodeError:
                decoded_string[channel] = "The string could not be decoded using utf-8."

            for saved_string in saved_strings:
                if len(decoded_string[channel]) < len(saved_string):
                    smaller_string = decoded_string[channel]
                    larger_string = saved_string
                else:
                    smaller_string = saved_string
                    larger_string = decoded_string[channel]

                equal_characters = sum(c1 == c2 for c1, c2 in zip(
                    smaller_string, larger_string))

                if equal_characters / len(smaller_string) >= 0.5:
                    comparison_result[channel] = True
                else:
                    comparison_result[channel] = False

        return comparison_result

    @staticmethod
    def psnr(original, compressed):
        compressed = PVDText.resize(original, compressed)

        mse_total = 0

        if len(original.shape) == 3 and len(compressed.shape) == 3:
            mse_r = np.mean((original[:, :, 0] - compressed[:, :, 0]) ** 2)
            mse_g = np.mean((original[:, :, 1] - compressed[:, :, 1]) ** 2)
            mse_b = np.mean((original[:, :, 2] - compressed[:, :, 2]) ** 2)

            mse_total = (mse_r + mse_g + mse_b) / 3

        elif len(original.shape) == 2 and len(compressed.shape) == 2:
            mse_total = np.mean((original - compressed) ** 2)

        if mse_total == 0:
            return float('inf')

        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_total))

        return psnr_value

    @staticmethod
    def resize(reference_image, image_to_resize):
        reference_height = reference_image.shape[0]
        reference_width = reference_image.shape[1]
        resized_image = cv2.resize(
            image_to_resize, (reference_width, reference_height))

        return resized_image
