from flask import flash
import os
import cv2
import numpy as np
import uuid
from invisimark.services.dct_image import DCTImage
from invisimark.services.dct_text import DCTText
from invisimark.services.dwt_image import DWTImage
from invisimark.services.dwt_text import DWTText
from invisimark.services.lsb_image import LSBImage
from invisimark.services.lsb_text import LSBText
from invisimark.services.hs_image import HSText
from invisimark.services.pvd_image import PVDImage
from invisimark.services.pvd_text import PVDText

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
USERS_IMAGES = os.path.join(REPO_DIR, 'images')
WATERMARKS_PATH = os.path.join(USERS_IMAGES, 'watermarks')

class ImageProcessor:
    def __init__(self, app):
        self.app = app

    @staticmethod
    def convertNpArray(input, path_config):
        file = cv2.imdecode(np.fromstring(
            open(os.path.join(path_config, input), 'rb').read(), np.uint8), cv2.IMREAD_UNCHANGED)
        return file

    @staticmethod
    def convert_filestorage_to_numpy_array(filestorage):
        file_bytes = filestorage.read()
        image_array = cv2.imdecode(np.frombuffer(
            file_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image_array

    @staticmethod
    def verify_extract(watermark, extracted_watermark):
      if isinstance(extracted_watermark, np.ndarray) and extracted_watermark.ndim == 3:
        correlation = ImageProcessor.calculate_watermark_correlation(
            watermark, extracted_watermark)

        return correlation, extracted_watermark
      else:
          correlations = [ImageProcessor.calculate_watermark_correlation(
              watermark, block) for block in extracted_watermark]
          max_index, max_correlation = max(
              enumerate(correlations), key=lambda x: x[1])
          watermark_with_max_correlation = extracted_watermark[max_index]

          return max_correlation, watermark_with_max_correlation
    @staticmethod
    def calculate_watermark_correlation(watermark_input, extracted_watermark):
        extracted_watermark = DCTImage.resize(
            watermark_input, extracted_watermark)

        if watermark_input.shape != extracted_watermark.shape:
            raise ValueError(
                "As imagens das marcas d'água têm tamanhos diferentes.")

        input_mean = np.mean(watermark_input)
        extracted_mean = np.mean(extracted_watermark)

        input_diff = watermark_input - input_mean
        extracted_diff = extracted_watermark - extracted_mean

        term1 = np.sum(input_diff * extracted_diff)
        term2 = np.sqrt(np.sum(input_diff ** 2) * np.sum(extracted_diff ** 2))

        correlation = term1 / term2

        return correlation

    @staticmethod
    def calculate_psnr(original, compressed):
        compressed = DCTImage.resize(original, compressed)

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
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse_total))

        return psnr

    @staticmethod
    def perform_insertion(original_image, insertion_type, watermark=None, text=None):
        if insertion_type == 'image_dct':
            marked_image = DCTImage.rgb_insert_dct(
                original_image, watermark)
        elif insertion_type == 'text_dct':
            marked_image = DCTText.rgb_insert_text_dct(original_image, text)
        elif insertion_type == 'image_dwt':
            marked_image = DWTImage.embed_watermark_HH_blocks(
                original_image, watermark)
        elif insertion_type == 'text_dwt':
            marked_image = DWTText.embed_text_watermark(original_image, text)
        elif insertion_type == 'image_lsb':
            marked_image = LSBImage.embed_LSB_image(
                original_image, watermark)
        elif insertion_type == 'text_lsb':
            marked_image = LSBText.encode(original_image, text)
        elif insertion_type == 'text_hs':
            marked_image = HSText.encode_HS(
                original_image, text)
        elif insertion_type == 'image_pvd':
            marked_image = PVDImage.hide_image_pvd(
                original_image, watermark)
        elif insertion_type == 'text_pvd':
            marked_image = PVDText.hide_message(original_image, text)
        else:
            flash('Tipo de inserção não suportado.', 'danger')
            return None

        return marked_image

    @staticmethod
    def perform_extraction(original_image, marked_image, extraction_type, watermark=None, text=None):
        marked_image = DCTImage.resize(original_image, marked_image)

        if extraction_type == 'image_dct':
            watermark = DCTImage.rgb_remove_dct(original_image, marked_image)
        elif extraction_type == 'text_dct':
            watermark = DCTText.rgb_extract_text_dct(
                original_image, marked_image, len(text))
        elif extraction_type == 'image_dwt':
            watermark = DWTImage.extract_watermark_HH_blocks(
                original_image, marked_image)
        elif extraction_type == 'text_dwt':
            watermark = DWTText.extract_text_watermark(
                original_image, marked_image, len(text))
        elif extraction_type == 'image_lsb':
            watermark = LSBImage.blind_extraction_LSB(
                marked_image)
        elif extraction_type == 'text_lsb':
            watermark = LSBText.extract(marked_image)
        elif extraction_type == 'text_hs':
            watermark = HSText.extract_HS(
                original_image, marked_image)
        elif extraction_type == 'image_pvd':
            watermark = PVDImage.extract_image_pvd(
                original_image, marked_image)
        elif extraction_type == 'text_pvd':
            watermark = PVDText.extract_message(
                original_image, marked_image)
        else:
            flash('Tipo de extração não suportado.', 'danger')
            return None

        return watermark

    @staticmethod
    def save_watermark_file(file):
        watermark_filename = os.path.join(
            WATERMARKS_PATH, f"{str(uuid.uuid4())}.png")
        file.save(watermark_filename)
        return watermark_filename
