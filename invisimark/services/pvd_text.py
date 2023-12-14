import cv2
import numpy as np
import os
import math

class PVDText:
    
    @staticmethod
    def hide_message(image, text):
        # Convert the string to a byte array.
        text_bytes = text.encode('utf-8')

        new_image = PVDText.compare_area_string(image, text_bytes)

        # Create an empty array to store the stego image.
        stego_image_r, secret_message_r = PVDText.hide_message_rgb(new_image, text_bytes, 'r')
        stego_image_g, secret_message_g = PVDText.hide_message_rgb(new_image, text_bytes, 'g')
        stego_image_b, secret_message_b = PVDText.hide_message_rgb(new_image, text_bytes, 'b')

        # Calculate PSNR for each channel.
        psnr_r = PVDText.psnr(image, stego_image_r)
        psnr_g = PVDText.psnr(image, stego_image_g)
        psnr_b = PVDText.psnr(image, stego_image_b)

        # Choose the channel with the best PSNR.
        best_channel = np.argmax([psnr_r, psnr_g, psnr_b])

        # Combine the bits of the secret message with the least significant bits of the chosen RGB channel.
        stego_image = stego_image_r if best_channel == 0 else stego_image_g if best_channel == 1 else stego_image_b
        secret_message = secret_message_r if best_channel == 0 else secret_message_g if best_channel == 1 else secret_message_b
        print(secret_message)

        return stego_image, secret_message

    @staticmethod
    def hide_message_rgb(cover_image, message, rgb):
        """
        Hides a message within an image using the PVD technique in a single channel.

        Args:
            cover_image: The cover image.
            message: The message to be hidden.
            rgb: The color channel to be modified ('r', 'g', or 'b').

        Returns:
            The stego image and the secret message.
        """
        # Create a copy of the cover image to store the result.
        stego_image = np.copy(cover_image)

        # Check if the message fits in the image.
        cover_image = PVDText.compare_area_string(cover_image, message)

        # Separate the color channels of the cover image.
        cover_b, cover_g, cover_r = cv2.split(cover_image)
        # Convert the message to binary.

        # Choose the color channel of the cover image based on the rgb argument.
        if rgb == 'r':
            cover_channel = cover_r
        elif rgb == 'g':
            cover_channel = cover_g
        elif rgb == 'b':
            cover_channel = cover_b
        else:
            raise ValueError("rgb must be 'r', 'g', or 'b'")

        # Iterate over each bit in the message.
        for i in range(len(message)):
            # Get the current bit of the message.
            message_bit = int(message[i])

            # Get the corresponding pixel in the cover image.
            y = i // cover_channel.shape[1]
            x = i % cover_channel.shape[1]
            pixel = cover_channel[y, x]

            # Clear the least significant bit of the pixel.
            pixel &= 0xFE

            # Replace the least significant bit of the pixel with the current bit of the message.
            pixel |= message_bit

            # Update the pixel in the cover image.
            cover_channel[y, x] = pixel

        # Combine the color channels to create the stego image.
        if rgb == 'r':
            stego_image = cv2.merge((cover_b, cover_g, cover_channel))
        elif rgb == 'g':
            stego_image = cv2.merge((cover_b, cover_channel, cover_r))
        elif rgb == 'b':
            stego_image = cv2.merge((cover_channel, cover_g, cover_r))

        return stego_image, message

    @staticmethod
    def extract_message(stego_image):
        """
        Extracts a message from an image using the PVD technique in all channels.

        Args:
            stego_image: The steganographic image.

        Returns:
            The secret messages from each channel.
        """
        # Separate the color channels of the steganographic image.
        stego_b, stego_g, stego_r = cv2.split(stego_image)

        # Create a dictionary to store the message bits for each channel.
        message_bits = {'r': [], 'g': [], 'b': []}

        # Iterate over each pixel in each channel of the steganographic image.
        for channel, stego_channel in zip(['r', 'g', 'b'], [stego_r, stego_g, stego_b]):
            for y in range(stego_channel.shape[0]):
                for x in range(stego_channel.shape[1]):
                    # Get the least significant bit of the pixel.
                    bit_message = stego_channel[y, x] & 0x01

                    # Add the bit to the message bits list.
                    message_bits[channel].append(bit_message)

        # Convert the list of bits to a binary string and then to a text string for each channel.
        message = {channel: ''.join(chr(int(''.join(str(bit) for bit in message_bits[channel][i:i+8]), 2)) for i in range(
            0, len(message_bits[channel]) - len(message_bits[channel]) % 8, 8)) for channel in ['r', 'g', 'b']}

        return message

    @staticmethod
    def compare_area_string(image, message):
        # Calculate the area of the image in pixels.
        image_area = image.shape[0] * image.shape[1]

        # Check if the area of the image is greater than or equal to the size of the byte array.
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
        """
        Compares the extracted message with all the strings saved in a file, character by character.

        Args:
            decoded_message: The extracted message.

        Returns:
            A dictionary indicating whether the extracted message matches any string in the file for each channel.
        """
        # Read all the strings from the file.
        file_path = os.path.join('watermarked_images', 'message.txt')
        with open(file_path, 'r') as file:
            saved_strings = file.read().splitlines()

        # Create a dictionary to store the comparison results for each channel.
        comparison_result = {}

        # Create a dictionary to store the decoded messages for each channel.
        decoded_string = {}

        # Compare the extracted message with all the saved strings for each channel.
        for channel in ['r', 'g', 'b']:
            # Decode the byte-escaped string back to a string.
            try:
                decoded_string[channel] = bytes(decoded_message[channel], 'latin1').decode('utf-8')
            except UnicodeDecodeError:
                decoded_string[channel] = "The string could not be decoded using utf-8."

            for saved_string in saved_strings:
                # Determine the smaller and larger strings.
                if len(decoded_string[channel]) < len(saved_string):
                    smaller_string = decoded_string[channel]
                    larger_string = saved_string
                else:
                    smaller_string = saved_string
                    larger_string = decoded_string[channel]

                # Compare the strings character by character.
                equal_characters = sum(c1 == c2 for c1, c2 in zip(smaller_string, larger_string))

                # Check if at least 50% of characters are equal.
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
            # Assuming original and compressed are RGB color images
            mse_r = np.mean((original[:, :, 0] - compressed[:, :, 0]) ** 2)
            mse_g = np.mean((original[:, :, 1] - compressed[:, :, 1]) ** 2)
            mse_b = np.mean((original[:, :, 2] - compressed[:, :, 2]) ** 2)

            # Calculate the average of MSEs for R, G, and B channels
            mse_total = (mse_r + mse_g + mse_b) / 3

        elif len(original.shape) == 2 and len(compressed.shape) == 2:
            # If images are grayscale (single channel)
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