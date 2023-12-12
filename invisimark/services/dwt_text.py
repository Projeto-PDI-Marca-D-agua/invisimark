import cv2
import pywt
import numpy as np


class DWTText:
    @staticmethod
    def embed_text_watermark(original_image, text):
        alpha = 0.1
        b, g, r = cv2.split(original_image)

        coeffs_b = pywt.dwt2(b, 'haar')
        coeffs_g = pywt.dwt2(g, 'haar')
        coeffs_r = pywt.dwt2(r, 'haar')

        LL_b, _ = coeffs_b
        LL_g, _ = coeffs_g
        LL_r, _ = coeffs_r

        ascii_values = DWTText.text_to_ascii(text)

        original_text_length = len(ascii_values)

        ascii_b = np.resize(ascii_values, LL_b.shape)
        ascii_g = np.resize(ascii_values, LL_g.shape)
        ascii_r = np.resize(ascii_values, LL_r.shape)

        LL_b += alpha * ascii_b
        LL_g += alpha * ascii_g
        LL_r += alpha * ascii_r

        watermarked_b = pywt.idwt2((LL_b, coeffs_b[1]), 'haar')
        watermarked_g = pywt.idwt2((LL_g, coeffs_g[1]), 'haar')
        watermarked_r = pywt.idwt2((LL_r, coeffs_r[1]), 'haar')

        watermarked_image = cv2.merge(
            (watermarked_b, watermarked_g, watermarked_r))

        # return original_text_length
        return watermarked_image

    @staticmethod
    def extract_text_watermark(original_image, marked_image, original_text_length):
        alpha = 0.1
        b, g, r = cv2.split(marked_image)
        b_original, g_original, r_original = cv2.split(original_image)

        coeffs_original_b = pywt.dwt2(b_original, 'haar')
        coeffs_original_g = pywt.dwt2(g_original, 'haar')
        coeffs_original_r = pywt.dwt2(r_original, 'haar')

        coeffs_watermarked_b = pywt.dwt2(b, 'haar')
        coeffs_watermarked_g = pywt.dwt2(g, 'haar')
        coeffs_watermarked_r = pywt.dwt2(r, 'haar')

        LL_original_b, _ = coeffs_original_b
        LL_watermarked_b, _ = coeffs_watermarked_b
        watermark_extracted_b = (LL_watermarked_b - LL_original_b) / alpha

        watermark_text_b = DWTText.ascii_to_text(
            watermark_extracted_b.flatten()[:original_text_length])

        return watermark_text_b

    @staticmethod
    def text_to_ascii(text):
        return [ord(char) for char in text]

    @staticmethod
    def ascii_to_text(ascii_values):
        return ''.join([chr(int(val)) for val in ascii_values])
