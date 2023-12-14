import cv2 as cv
import cv2
import numpy as np


class HSText:
    @staticmethod
    def string_to_bitstream(message):
        """ Convert a string to a list of binary strings """
        return [bin(ord(character))[2:].zfill(8) for character in message]

    @staticmethod
    def bitstream_to_string(bit_stream):
        """ Convert a list of binary strings to a string """
        return ''.join(chr(int(''.join(bit_stream[i:i+8]), 2)) for i in range(0, len(bit_stream), 8))

    @staticmethod
    def histogram(image):
        num_bits = 256
        intensities_array = np.zeros((3, num_bits))
        img_height = image.shape[0]
        img_width = image.shape[1]

        for i in range(img_height):
            for j in range(img_width):
                for k in range(3):  # Loop for the 3 color channels
                    pixel_value = image[i][j][k]
                    intensities_array[k][pixel_value] += 1

        x = np.arange(num_bits)

        return x, intensities_array

    @staticmethod
    def encode_HS(img_name, text):
        """ Method to hide a message in an 8-bit color image """
        cover_image = img_name

        # Convert message to bitstream
        bit_stream = ''.join(HSText.string_to_bitstream(text + '$$'))

        bins, intensities = HSText.histogram(cover_image)
        peak_point = np.argmax(intensities)

        # Check data stream size
        size = len(bit_stream)

        # Shift histogram values to the right larger than the peak by 1
        img_height = cover_image.shape[0]
        img_width = cover_image.shape[1]
        for i in range(img_height):
            for j in range(img_width):
                for k in range(3):  # Loop for the 3 color channels
                    pixel = cover_image[i][j][k]
                    if peak_point < pixel < len(bins) - 1:
                        cover_image[i][j][k] += 1

        # Hide the message
        bit_count = 0
        for i in range(img_height):
            for j in range(img_width):
                for k in range(3):  # Loop for the 3 color channels
                    if bit_count < size:
                        if cover_image[i][j][k] == peak_point:
                            if bit_stream[bit_count] == '1':
                                cover_image[i][j][k] += 1
                            bit_count += 1
                    else:
                        break

        return cover_image

    @staticmethod
    def extract_HS(img_name, img_name2):
        """ Method to reveal a bitstream in an 8-bit color image """

        marked_image = img_name

        original_image = img_name2

        img_height = marked_image.shape[0]
        img_width = marked_image.shape[1]

        bins, intensities = HSText.histogram(original_image)
        peak_point = np.argmax(intensities)

        # Get bitstream from the image
        bit_stream = []
        for i in range(img_height):
            for j in range(img_width):
                for k in range(3):  # Loop for the 3 color channels
                    pixel = marked_image[i][j][k]
                    if pixel == peak_point:
                        bit_stream.append('0')
                    elif pixel == peak_point + 1:
                        bit_stream.append('1')

        message_returned = HSText.bitstream_to_string(bit_stream)

        # Find '$$' (which marks the end of the secret message)
        termination_index = message_returned.find('$$')
        if termination_index != -1:
            message_returned = message_returned[:termination_index]

        return message_returned
