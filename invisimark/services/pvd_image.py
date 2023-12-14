import cv2
import numpy as np
import math


class PVDImage:
    @staticmethod
    def hide_image_pvd(cover_image, secret_image):
        # Check if the images have the same size.
        if cover_image.shape != secret_image.shape:
            # If the images don't have the same size, resize the secret image.
            secret_image = PVDImage.resize(cover_image, secret_image)

        stego_image_r, secret_image_r = PVDImage.hide_image_rgb_channel_pvd(
            cover_image, secret_image, 'r')
        stego_image_g, secret_image_g = PVDImage.hide_image_rgb_channel_pvd(
            cover_image, secret_image, 'g')
        stego_image_b, secret_image_b = PVDImage.hide_image_rgb_channel_pvd(
            cover_image, secret_image, 'b')

        # Choose the stego image with the best PSNR.
        psnr_r = PVDImage.psnr(cover_image, stego_image_r)
        psnr_g = PVDImage.psnr(cover_image, stego_image_g)
        psnr_b = PVDImage.psnr(cover_image, stego_image_b)
        best_channel = np.argmax([psnr_r, psnr_g, psnr_b])

        # Combine the bits of the secret image with the least significant bits of the chosen RGB channel.
        stego_image = stego_image_r if best_channel == 0 else stego_image_g if best_channel == 1 else stego_image_b
        secret_image = secret_image_r if best_channel == 0 else secret_image_g if best_channel == 1 else secret_image_b

        return stego_image, secret_image

    @staticmethod
    def hide_image_rgb_channel_pvd(cover_image, secret_image, rgb):
        # Check if the cover image is a single channel (grayscale)
        if len(cover_image.shape) < 3 or cover_image.shape[2] == 1:
            return "The message cannot be encoded because the cover image is a single channel."

        # Create a copy of the cover image to store the result.
        stego_image = np.copy(cover_image)

        # Separate the color channels of the cover and secret images.
        cover_b, cover_g, cover_r = cv2.split(cover_image)
        secret_b, secret_g, secret_r = cv2.split(secret_image)

        # Choose the color channel of the cover and secret images based on the rgb argument.
        if rgb == 'r':
            cover_channel = cover_r
            secret_channel = secret_r
        elif rgb == 'g':
            cover_channel = cover_g
            secret_channel = secret_g
        elif rgb == 'b':
            cover_channel = cover_b
            secret_channel = secret_b
        else:
            raise ValueError("rgb must be 'r', 'g', or 'b'")

        # Iterate over each pixel in the chosen channel of the secret image.
        for y in range(secret_channel.shape[0]):
            for x in range(secret_channel.shape[1]):
                # Get the 4 most significant bits of the pixel in the chosen channel of the secret image.
                most_significant_bits = (secret_channel[y, x] >> 4) & 0x0F

                # Clear the 4 least significant bits of the pixel in the chosen channel of the cover image.
                cover_channel[y, x] &= 0xF0

                # Replace the 4 least significant bits of the pixel in the chosen channel of the cover image with the 4 most significant bits of the pixel in the chosen channel of the secret image.
                cover_channel[y, x] |= most_significant_bits

        # Combine the color channels to create the stego image.
        if rgb == 'r':
            stego_image = cv2.merge((cover_b, cover_g, cover_channel))
        elif rgb == 'g':
            stego_image = cv2.merge((cover_b, cover_channel, cover_r))
        elif rgb == 'b':
            stego_image = cv2.merge((cover_channel, cover_g, cover_r))

        return stego_image, secret_channel

    @staticmethod
    def extract_image_pvd(stego_image):
        """
        Extracts a hidden image from another using the PVD technique in all channels.

        Args:
        stego_image: The stego image.

        Returns:
        The extracted secret images from each channel.
        """
        # Create an empty array to store the extracted secret images.
        secret_image_r = np.zeros_like(stego_image)
        secret_image_g = np.zeros_like(stego_image)
        secret_image_b = np.zeros_like(stego_image)

        # Set quantization levels.
        levels = np.array([0, 16, 32, 48, 64, 80, 96, 112,
                          128, 144, 160, 176, 192, 208, 224, 240, 255])

        # Check if the stego image is grayscale.
        if len(stego_image.shape) == 2:
            # If the stego image is grayscale, extract the secret image from the gray channel.
            secret_image_gray = np.zeros_like(stego_image)
            channels = [('gray', stego_image, secret_image_gray)]
            return secret_image_gray
        else:
            # If the stego image is colored, extract the secret image from the R, G, and B channels.
            stego_b, stego_g, stego_r = cv2.split(stego_image)
            channels = [('r', stego_r, secret_image_r), ('g', stego_g,
                                                         secret_image_g), ('b', stego_b, secret_image_b)]

        # Iterate over each pixel in each channel of the stego image.
        channel_name_r, stego_r, secret_image_r = channels[0]
        for y in range(stego_r.shape[0]):
            for x in range(stego_r.shape[1]):
                # Extract the 4 least significant bits of the pixel in the chosen channel of the stego image.
                least_significant_bits = stego_r[y, x] & 0x0F

                # Move the 4 least significant bits to the position of the 4 most significant bits.
                least_significant_bits <<= 4

                index = np.argmin(np.abs(levels - least_significant_bits))
                least_significant_bits = levels[index]

                # Store the extracted bits in the corresponding secret image.
                secret_image_r[y, x] = least_significant_bits
                secret_image_r[y, x] &= 0xF0
                secret_image_r[y, x] |= int(least_significant_bits / 16)

        channel_name_g, stego_g, secret_image_g = channels[1]
        for y in range(stego_g.shape[0]):
            for x in range(stego_g.shape[1]):
                # Extract the 4 least significant bits of the pixel in the chosen channel of the stego image.
                least_significant_bits = stego_g[y, x] & 0x0F

                # Move the 4 least significant bits to the position of the 4 most significant bits.
                least_significant_bits <<= 4

                index = np.argmin(np.abs(levels - least_significant_bits))
                least_significant_bits = levels[index]

                # Store the extracted bits in the corresponding secret image.
                secret_image_g[y, x] = least_significant_bits
                secret_image_g[y, x] &= 0xF0
                secret_image_g[y, x] |= int(least_significant_bits / 16)

        channel_name_b, stego_b, secret_image_b = channels[2]
        for y in range(stego_b.shape[0]):
            for x in range(stego_b.shape[1]):
                # Extract the 4 least significant bits of the pixel in the chosen channel of the stego image.
                least_significant_bits = stego_b[y, x] & 0x0F

                # Move the 4 least significant bits to the position of the 4 most significant bits.
                least_significant_bits <<= 4

                index = np.argmin(np.abs(levels - least_significant_bits))
                least_significant_bits = levels[index]

                # Store the extracted bits in the corresponding secret image.
                secret_image_b[y, x] = least_significant_bits
                secret_image_b[y, x] &= 0xF0
                secret_image_b[y, x] |= int(least_significant_bits / 16)

        return secret_image_r, secret_image_g, secret_image_b


    @staticmethod
    def resize(reference_image, image_to_resize):
        reference_height = reference_image.shape[0]
        reference_width = reference_image.shape[1]
        resized_image = cv2.resize(
            image_to_resize, (reference_width, reference_height))

        return resized_image

    @staticmethod
    def calculate_watermark_correlation(input_watermark, extracted_watermark):
        extracted_watermark = PVDImage.resize(
            input_watermark, extracted_watermark)

        # Check if both images have the same size
        if input_watermark.shape != extracted_watermark.shape:
            raise ValueError("Watermark images have different sizes")

        # Calculate the mean pixel intensities for both images
        mean_input = np.mean(input_watermark)
        mean_extracted = np.mean(extracted_watermark)

        # Calculate the difference between pixel intensities and the mean
        diff_input = input_watermark - mean_input
        diff_extracted = extracted_watermark - mean_extracted

        # Calculate terms for the normalized cross-correlation formula
        term1 = np.sum(diff_input * diff_extracted)
        term2 = np.sqrt(np.sum(diff_input ** 2) * np.sum(diff_extracted ** 2))

        # Calculate normalized cross-correlation
        correlation = term1 / term2

        return correlation

    @staticmethod
    def psnr(original, compressed):
        compressed = PVDImage.resize(original, compressed)

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
