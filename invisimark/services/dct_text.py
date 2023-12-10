import numpy as np
import cv2


class DCTText:
    @staticmethod
    def rgb_insert_texto(original_image, text, alpha):
        # Verificando as dimensões da imagem de marca d'água
        height, width, _ = original_image.shape

        # Transformando o texto em valores ASCII
        text_array = np.array([ord(char) for char in text])

        # Convertendo a imagem para ponto flutuante
        original_image_float = original_image.astype(np.float32)

        # Divide a imagem em canais de cor
        channel_blue, channel_green, channel_red = cv2.split(
            original_image_float)

        # Aplicando a DCT a cada canal de cor
        dct_blue = cv2.dct(channel_blue)
        dct_green = cv2.dct(channel_green)
        dct_red = cv2.dct(channel_red)

        # Inserindo o texto até ele acabar ou até a imagem acabar
        for i in range(min(len(text_array), original_image.size)):
            dct_blue[i // width, i % width] += alpha * text_array[i]

        # Aplicando a IDCT para obter a imagem marcada
        inverse_blue = cv2.idct(dct_blue)
        inverse_green = cv2.idct(dct_green)
        inverse_red = cv2.idct(dct_red)

        # Combinando os canais para obter a imagem final
        marked_image = cv2.merge((inverse_blue, inverse_green, inverse_red))

        # Convertendo a imagem de volta para uint8, se necessário
        marked_image = marked_image.astype(np.uint8)

        return marked_image

    @staticmethod
    def rgb_extract_text(marked_image, original_image, alpha, text_length):
        # Dividindo a imagem em canais de cor
        channel_blue_marked, _, _ = cv2.split(marked_image)
        channel_blue_original, _, _ = cv2.split(original_image)

        # Calculando a DCT 2D da imagem marcada
        blue_marked_dct = cv2.dct(np.float32(channel_blue_marked))

        # Calculando a DCT 2D da imagem original
        blue_original_dct = cv2.dct(np.float32(channel_blue_original))

        # Inicializando um array para armazenar os valores ASCII extraídos
        extracted_text = []

        # Iterando sobre os coeficientes DCT que contêm o texto
        for i in range(min(text_length, blue_marked_dct.size)):
            # Subtrai a DCT da imagem original da DCT da imagem marcada
            diff_coefficient = blue_marked_dct[i // blue_marked_dct.shape[1], i % blue_marked_dct.shape[1]
                                               ] - blue_original_dct[i // blue_original_dct.shape[1], i % blue_original_dct.shape[1]]
            extracted_text.append(diff_coefficient / alpha)

        # Converte os valores ASCII em uma string
        extracted_text = ''.join([chr(int(round(value)) % 256)
                                 for value in extracted_text])

        return extracted_text
