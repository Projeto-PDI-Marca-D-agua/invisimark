import cv2
import numpy as np
# from PIL import Image
import random


class LSBText:
    @staticmethod
    def convert_data_to_binary(data):
        if type(data) == str:
            binary_data = ''.join([format(ord(i), '08b') for i in data])
        elif type(data) == bytes or type(data) == np.ndarray:
            binary_data = [format(i, '08b') for i in data]
        return binary_data

    @staticmethod
    def hide_text(img, data):
        data_index = 0
        # '##' at the beginning of the message
        data = '##' + data + '$$'
        binary_data = LSBText.convert_data_to_binary(data)
        len_data = len(binary_data)
        width, height, _ = img.shape

        # Divide the image into blocks
        block_size = 100
        blocks = [img[i:i+block_size, j:j+block_size]
                  for i in range(0, width, block_size) for j in range(0, height, block_size)]

        # Randomly select half of the blocks
        num_blocks = len(blocks)
        selected_blocks = random.sample(blocks, num_blocks // 2)

        for block in selected_blocks:
            rows, cols, _ = block.shape

            for i in range(rows):
                for j in range(cols):
                    if data_index < len_data:
                        for k in range(3):  # Red, Green, Blue channels
                            if data_index < len_data:
                                block[i, j, k] = int(format(block[i, j, k], '08b')[
                                                     :-1] + binary_data[data_index], 2)
                                data_index += 1
                    if data_index >= len_data:
                        break

        return img

    @staticmethod
    def encode(original_image, text):
        image = original_image

        if len(text) == 0:
            raise ValueError("Empty data")

        width, height, _ = image.shape

        # Hide the data in the image
        img = image.copy()
        img = LSBText.hide_text(img, text)

        return img

    @staticmethod
    def search_text(img):
        width, height, _ = img.shape
        found_data = False

        # Divide the image into blocks
        block_size = 100
        blocks = [img[i:i+block_size, j:j+block_size]
                  for i in range(0, width, block_size) for j in range(0, height, block_size)]

        for block in blocks:
            binary_data = ""
            rows, cols, _ = block.shape

            for i in range(rows):
                for j in range(cols):
                    for k in range(3):
                        binary_data += format(block[i, j, k], '08b')[-1]

            all_bytes = [binary_data[i: i + 8]
                         for i in range(0, len(binary_data), 8)]

            readable_data = ""
            data_started = False

            for i in range(len(all_bytes) - 1):
                chars = chr(int(all_bytes[i], 2)) + chr(int(all_bytes[i+1], 2))
                if chars == '##' and not data_started:
                    data_started = True
                    i += 1  # Next byte because we've already processed it
                elif data_started:
                    readable_data += chr(int(all_bytes[i], 2))
                    if readable_data[-2:] == "$$":
                        readable_data = readable_data[1:-2]
                        found_data = True
                        break
            if found_data:
                break

        return readable_data

    @staticmethod
    def extract(marked_image):
        image = marked_image
        decoded_data = LSBText.search_text(image)

        end_marker = decoded_data.find("$$")

        if end_marker != -1:
            decoded_data = decoded_data[:end_marker]

        return decoded_data
