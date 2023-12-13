import cv2
import pywt


class DWTImage:
    @staticmethod
    def embed_watermark_HH_blocks(image, watermark):
        alpha = 0.1
        b, g, r = cv2.split(image)

        coeffs_b = pywt.dwt2(b, 'haar')
        coeffs_g = pywt.dwt2(g, 'haar')
        coeffs_r = pywt.dwt2(r, 'haar')

        LL_b, (LH_b, HL_b, HH_b) = coeffs_b
        LL_g, (LH_g, HL_g, HH_g) = coeffs_g
        LL_r, (LH_r, HL_r, HH_r) = coeffs_r

        watermark_b, watermark_g, watermark_r = cv2.split(watermark)
        watermark_b_resized = cv2.resize(
            watermark_b, (HH_b.shape[1]//3, HH_b.shape[0]//3))
        watermark_g_resized = cv2.resize(
            watermark_g, (HH_g.shape[1]//3, HH_g.shape[0]//3))
        watermark_r_resized = cv2.resize(
            watermark_r, (HH_r.shape[1]//3, HH_r.shape[0]//3))

        for i in range(3):
            for j in range(3):
                HH_b[i * (HH_b.shape[0]//3): (i + 1) * (HH_b.shape[0]//3), j * (
                    HH_b.shape[1]//3): (j + 1) * (HH_b.shape[1]//3)] += alpha * watermark_b_resized
                HH_g[i * (HH_g.shape[0]//3): (i + 1) * (HH_g.shape[0]//3), j * (
                    HH_g.shape[1]//3): (j + 1) * (HH_g.shape[1]//3)] += alpha * watermark_g_resized
                HH_r[i * (HH_r.shape[0]//3): (i + 1) * (HH_r.shape[0]//3), j * (
                    HH_r.shape[1]//3): (j + 1) * (HH_r.shape[1]//3)] += alpha * watermark_r_resized

        watermarked_b = pywt.idwt2((LL_b, (LH_b, HL_b, HH_b)), 'haar')
        watermarked_g = pywt.idwt2((LL_g, (LH_g, HL_g, HH_g)), 'haar')
        watermarked_r = pywt.idwt2((LL_r, (LH_r, HL_r, HH_r)), 'haar')

        watermarked_image = cv2.merge(
            (watermarked_b, watermarked_g, watermarked_r))

        return watermarked_image

    @staticmethod
    def extract_watermark_HH_blocks(original_image, marked_image):
        alpha = 0.1
        
        b, g, r = cv2.split(marked_image)
        b_original, g_original, r_original = cv2.split(original_image)

        coeffs_original_b = pywt.dwt2(b_original, 'haar')
        coeffs_original_g = pywt.dwt2(g_original, 'haar')
        coeffs_original_r = pywt.dwt2(r_original, 'haar')

        coeffs_watermarked_b = pywt.dwt2(b, 'haar')
        coeffs_watermarked_g = pywt.dwt2(g, 'haar')
        coeffs_watermarked_r = pywt.dwt2(r, 'haar')

        LL_bW, (LH_bW, HL_bW, HH_bW) = coeffs_watermarked_b
        LL_gW, (LH_gW, HL_gW, HH_gW) = coeffs_watermarked_g
        LL_rW, (LH_rW, HL_rW, HH_rW) = coeffs_watermarked_r

        LL_b, (LH_b, HL_b, HH_b) = coeffs_original_b
        LL_g, (LH_g, HL_g, HH_g) = coeffs_original_g
        LL_r, (LH_r, HL_r, HH_r) = coeffs_original_r

        watermark_extracted_b = (HH_bW - HH_b) / alpha
        watermark_extracted_g = (HH_gW - HH_g) / alpha
        watermark_extracted_r = (HH_rW - HH_r) / alpha

        watermark_extracted = cv2.merge(
            (watermark_extracted_b, watermark_extracted_g, watermark_extracted_r))

        block_size_x = watermark_extracted.shape[0] // 3
        block_size_y = watermark_extracted.shape[1] // 3

        extracted_watermarks = []

        for i in range(3):
            for j in range(3):
                block = watermark_extracted[i * block_size_x: (
                    i + 1) * block_size_x, j * block_size_y: (j + 1) * block_size_y]
                extracted_watermarks.append(block)

        return extracted_watermarks

    @staticmethod
    def resize(original_image, watermark):
        return cv2.resize(watermark, (original_image.shape[1], original_image.shape[0]))
