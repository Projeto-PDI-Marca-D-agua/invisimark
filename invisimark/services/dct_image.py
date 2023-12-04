import numpy as np
import cv2

class DCTImage:
    @staticmethod
    def rgb_insert_dct_blocks(original_image, watermark, alpha):
        # Dividindo a imagem em canais de cor
        channel_blue, channel_green, channel_red = cv2.split(original_image)

        block_size_x = original_image.shape[0] // 3
        block_size_y = original_image.shape[1] // 3
        watermark = cv2.resize(watermark, (block_size_y, block_size_x))

        # Aplicando DCT a cada canal de cor
        dct_blue = cv2.dct(np.float32(channel_blue))
        dct_green = cv2.dct(np.float32(channel_green))
        dct_red = cv2.dct(np.float32(channel_red))

        # Ajustando a marca d'água para os canais de cor
        watermark_blue = alpha * watermark[:, :, 0]
        watermark_green = alpha * watermark[:, :, 1]
        watermark_red = alpha * watermark[:, :, 2]

        # Inserindo a marca d'água em cada bloco diretamente na DCT
        for i in range(3):
            for j in range(3):
                x_start, x_end = i * block_size_x, (i + 1) * block_size_x
                y_start, y_end = j * block_size_y, (j + 1) * block_size_y

                dct_blue[x_start:x_end, y_start:y_end] += watermark_blue
                dct_green[x_start:x_end, y_start:y_end] += watermark_green
                dct_red[x_start:x_end, y_start:y_end] += watermark_red

        # Aplicando IDCT para obter a imagem marcada
        inverse_blue = cv2.idct(dct_blue)
        inverse_green = cv2.idct(dct_green)
        inverse_red = cv2.idct(dct_red)

        # Combinando os canais para obter a imagem final
        marked_image = cv2.merge((inverse_blue, inverse_green, inverse_red))

        return marked_image

    @staticmethod
    def rgb_remove_dct_blocks(marked_image, original_image, alpha):
        height_reference, width_reference, _ = marked_image.shape

        original_image = cv2.resize(original_image, (width_reference, height_reference))

        channel_blue_marked, channel_green_marked, channel_red_marked = cv2.split(marked_image)

        channel_blue_marked = cv2.dct(np.float32(channel_blue_marked))
        channel_green_marked = cv2.dct(np.float32(channel_green_marked))
        channel_red_marked = cv2.dct(np.float32(channel_red_marked))

        channel_blue_original, channel_green_original, channel_red_original = cv2.split(original_image)

        channel_blue_original = cv2.dct(np.float32(channel_blue_original))
        channel_green_original = cv2.dct(np.float32(channel_green_original))
        channel_red_original = cv2.dct(np.float32(channel_red_original))

        watermark_blue = (channel_blue_marked - channel_blue_original) / alpha
        watermark_green = (channel_green_marked - channel_green_original) / alpha
        watermark_red = (channel_red_marked - channel_red_original) / alpha

        watermark_extracted = cv2.merge((watermark_blue, watermark_green, watermark_red))

        block_size_x = watermark_extracted.shape[0] // 3
        block_size_y = watermark_extracted.shape[1] // 3

        extracted_watermarks = []

        for i in range(3):
            for j in range(3):
                block = watermark_extracted[i * block_size_x: (i + 1) * block_size_x,
                        j * block_size_y: (j + 1) * block_size_y]
                extracted_watermarks.append(block)

        return extracted_watermarks
