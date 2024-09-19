from copy import deepcopy
from math import *
import matplotlib.pyplot as plt
import struct
import numpy as np


class read_file():

    f = open('img/img.bmp', "rb")
    header = f.read(54)
    bfType = header[:2]
    bfSize = header[2:6]
    bfReserved = header[6:10]
    bfOffBits = header[10:14]
    biSize = header[14:18]
    biWidth = header[18:22]
    biHeight = header[22:26]
    biPlanes = header[26:28]
    biBitCount = header[28:30]
    biCompression = header[30:34]
    biSizeImage = header[34:38]
    biXPelsPerMeter = header[38:42]
    biYPelsPerMeter = header[42:46]
    biClrUsed = header[46:50]
    biCrlImportant = header[50:54]
    pixels = []
    count = 1

    for i in range(int.from_bytes(biHeight, byteorder='little')):
        row = []
        for j in range(int.from_bytes(biWidth, byteorder='little')):
            b = ord(f.read(1))
            g = ord(f.read(1))
            r = ord(f.read(1))
            row.append([b, g, r])
        pixels.append(row)
    ost = f.read()
    f.close()

    def print_fields(self):
        print("Сигнатура:", self.bfType.decode('ascii'))
        print("Размер файла:", int.from_bytes(self.bfSize, byteorder='little'))
        print("Зарезервировано 1:", int.from_bytes(self.bfReserved, byteorder='little'))
        # print("Зарезервировано 2:", self.reserved2)
        print("Смещение данных:", int.from_bytes(self.bfOffBits, byteorder='little'))
        print("Размер заголовка:", int.from_bytes(self.biSize, byteorder='little'))
        print("Ширина:", int.from_bytes(self.biWidth, byteorder='little'))
        print("Высота:", int.from_bytes(self.biHeight, byteorder='little'))
        print("Число плоскостей:", int.from_bytes(self.biPlanes, byteorder='little'))
        print("Бит на пиксель:", int.from_bytes(self.biBitCount, byteorder='little'))
        print("Сжатие:", int.from_bytes(self.biCompression, byteorder='little'))
        print("Размер изображения:", int.from_bytes(self.biSizeImage, byteorder='little'))
        print("X пикселей на метр:", int.from_bytes(self.biXPelsPerMeter, byteorder='little'))
        print("Y пикселей на метр:", int.from_bytes(self.biYPelsPerMeter, byteorder='little'))
        print("Количество используемых цветов:", int.from_bytes(self.biClrUsed, byteorder='little'))
        print("Важные цвета:", int.from_bytes(self.biCrlImportant, byteorder='little'))


class BMPHeader:
    def __init__(self, header_data):
        # Распаковываем данные заголовка BMP
        (self.signature,
         self.file_size,
         self.reserved1,
         self.reserved2,
         self.data_offset,
         self.header_size,
         self.width,
         self.height,
         self.planes,
         self.bits_per_pixel,
         self.compression,
         self.image_size,
         self.x_pixels_per_meter,
         self.y_pixels_per_meter,
         self.colors_used,
         self.colors_important) = struct.unpack('<2sIHHIIBBHHIIBBII',
                                                header_data[:54])

    def print_fields(self):
        print("Сигнатура:", self.signature.decode('ascii'))
        print("Размер файла:", self.file_size)
        print("Зарезервировано 1:", self.reserved1)
        print("Зарезервировано 2:", self.reserved2)
        print("Смещение данных:", self.data_offset)
        print("Размер заголовка:", self.header_size)
        print("Ширина:", self.width)
        print("Высота:", self.height)
        print("Число плоскостей:", self.planes)
        print("Бит на пиксель:", self.bits_per_pixel)
        print("Сжатие:", self.compression)
        print("Размер изображения:", self.image_size)
        print("X пикселей на метр:", self.x_pixels_per_meter)
        print("Y пикселей на метр:", self.y_pixels_per_meter)
        print("Количество используемых цветов:", self.colors_used)
        print("Важные цвета:", self.colors_important)

def read_bmp(filename):
    with open(filename, 'rb') as f:
        # Считываем заголовок BMP файла
        header_data = f.read(54)  # Заголовок BMP файла имеет длину 54 байта
        print(f"len(header_data) {len(header_data)}")
        # Создаем экземпляр структуры BMPHeader и распаковываем заголовок
        header = BMPHeader(header_data)

        # Читаем данные изображения
        img_data = f.read()

    return header, img_data

def write_file(pixels):
    f = open(f"img/{read_file.count} изображение.bmp", 'wb')
    read_file.count += 1
    f.write(read_file.bfType)
    f.write(read_file.bfSize)
    f.write(read_file.bfReserved)
    f.write(read_file.bfOffBits)
    f.write(read_file.biSize)
    f.write(read_file.biWidth)
    f.write(read_file.biHeight)
    f.write(read_file.biPlanes)
    f.write(read_file.biBitCount)
    f.write(read_file.biCompression)
    f.write(read_file.biSizeImage)
    f.write(read_file.biXPelsPerMeter)
    f.write(read_file.biYPelsPerMeter)
    f.write(read_file.biClrUsed)
    f.write(read_file.biCrlImportant)
    for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
        for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
            r = pixels[i][j][0]
            g = pixels[i][j][1]
            b = pixels[i][j][2]
            f.write(bytes([r]))
            f.write(bytes([g]))
            f.write(bytes([b]))
    f.write(read_file.ost)
    f.close()
def writeFile(pixels):
    f1 = open(f"img/{read_file.count} изображение.bmp", 'wb')
    f2 = open(f"img/{read_file.count + 1} изображение.bmp", 'wb')
    f3 = open(f"img/{read_file.count + 2} изображение.bmp", 'wb')
    f1.write(read_file.bfType)
    f1.write(read_file.bfSize)
    f1.write(read_file.bfReserved)
    f1.write(read_file.bfOffBits)
    f1.write(read_file.biSize)
    f1.write(read_file.biWidth)
    f1.write(read_file.biHeight)
    f1.write(read_file.biPlanes)
    f1.write(read_file.biBitCount)
    f1.write(read_file.biCompression)
    f1.write(read_file.biSizeImage)
    f1.write(read_file.biXPelsPerMeter)
    f1.write(read_file.biYPelsPerMeter)
    f1.write(read_file.biClrUsed)
    f1.write(read_file.biCrlImportant)

    f2.write(read_file.bfType)
    f2.write(read_file.bfSize)
    f2.write(read_file.bfReserved)
    f2.write(read_file.bfOffBits)
    f2.write(read_file.biSize)
    f2.write(read_file.biWidth)
    f2.write(read_file.biHeight)
    f2.write(read_file.biPlanes)
    f2.write(read_file.biBitCount)
    f2.write(read_file.biCompression)
    f2.write(read_file.biSizeImage)
    f2.write(read_file.biXPelsPerMeter)
    f2.write(read_file.biYPelsPerMeter)
    f2.write(read_file.biClrUsed)
    f2.write(read_file.biCrlImportant)

    f3.write(read_file.bfType)
    f3.write(read_file.bfSize)
    f3.write(read_file.bfReserved)
    f3.write(read_file.bfOffBits)
    f3.write(read_file.biSize)
    f3.write(read_file.biWidth)
    f3.write(read_file.biHeight)
    f3.write(read_file.biPlanes)
    f3.write(read_file.biBitCount)
    f3.write(read_file.biCompression)
    f3.write(read_file.biSizeImage)
    f3.write(read_file.biXPelsPerMeter)
    f3.write(read_file.biYPelsPerMeter)
    f3.write(read_file.biClrUsed)
    f3.write(read_file.biCrlImportant)

    if 1 <= read_file.count <= 3:
        for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
            for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
                b = pixels[i][j][0]
                f1.write(bytes([b]))
                f1.write(bytes([0]))
                f1.write(bytes([0]))
        f1.write(read_file.ost)
        f1.close()
        for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
            for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
                g = pixels[i][j][1]
                f2.write(bytes([0]))
                f2.write(bytes([g]))
                f2.write(bytes([0]))
        f2.write(read_file.ost)
        f2.close()
        for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
            for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
                r = pixels[i][j][2]
                f3.write(bytes([0]))
                f3.write(bytes([0]))
                f3.write(bytes([r]))
        f3.write(read_file.ost)
        f3.close()
    elif 4 <= read_file.count <= 9:
        for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
            for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
                Y = pixels[i][j][0]
                f1.write(bytes([Y]))
                f1.write(bytes([Y]))
                f1.write(bytes([Y]))
        f1.write(read_file.ost)
        f1.close()
        for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
            for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
                Cb = pixels[i][j][1]
                f2.write(bytes([Cb]))
                f2.write(bytes([Cb]))
                f2.write(bytes([Cb]))
        f2.write(read_file.ost)
        f2.close()
        for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
            for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
                Cr = pixels[i][j][2]
                f3.write(bytes([Cr]))
                f3.write(bytes([Cr]))
                f3.write(bytes([Cr]))
        f3.write(read_file.ost)
        f3.close()
    read_file.count += 3


def ToYCbCr(pixels):
    YCbCr = []
    for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
        a = []
        for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
            Y = int(0.299 * pixels[i][j][2] + 0.587 * pixels[i][j][1] + 0.114 * pixels[i][j][0])
            Cb = int(0.5643 * (pixels[i][j][0] - Y) + 128)
            Cr = int(0.7132 * (pixels[i][j][2] - Y) + 128)
            if Y > 255:
                Y = 255
            if Cb > 255:
                Cb = 255
            if Cr > 255:
                Cr = 255
            a.append([Y, Cb, Cr])
        YCbCr.append(a)
    return YCbCr


def Mean(pixel, component_A):
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    summA = 0
    for i in range(H):
        for j in range(W):
            summA += pixel[i][j][component_A]
    return summA / (W*H)


# Среднее квадратичное отклонение компонент
def Deviation(pixel, A):
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    summA = 0
    MA = Mean(pixel, A)
    for i in range(H):
        for j in range(W):
            summA += (pixel[i][j][A] - MA ) ** 2
    return sqrt( summA / (W*H - 1) )
# Оценка коэффициента корреляции


# Оценка коэффициента корреляции
def CorelCoeff(pixel, component_A, component_B):
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    # Мат. ожидание
    MA = Mean(pixel, component_A)
    MB = Mean(pixel, component_B)
    znam = Deviation(pixel, component_A) * Deviation(pixel, component_B)
    tmp = 0
    for i in range(H):
        for j in range(W):
            tmp += (pixel[i][j][component_A] - MA) * (pixel[i][j][component_B] - MB)
    chisl = tmp / (W*H)
    return chisl / znam


def calc_autocorel_coeff(pixel, max_x,max_y, component_A):
    y = [-10, -5, 0, 5, 10]
    # x = [0] * max_x
    x = list(range(1, max_x + 1))
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')

    MA = Mean(pixel, component_A)
    # MB = Mean(pixel, component_B)
    znam = Deviation(pixel, component_A) * Deviation(pixel, component_A)
    tmp_ij = []
    tmp_mn = []

    for comp_y in y:
        for comp_x in x:
            for i in range(1, H-comp_y):
                for j in range(1, W - comp_x):
                    tmp_ij.append(pixel[i][j][component_A] - MA)

            for m in range(comp_y+1, H):
                for n in range(1, W - comp_x):
                    tmp_mn.append(pixel[m][n][component_A] - MA)


def calculateFirstSelection(colorArray, x, y, coeff):
    result = []
    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // coeff
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    if y >= 0:
        startValueI = 0
    else:
        startValueI = -y
    if y >= 0:
        endValueI = H - y
    else:
        endValueI = H
    if x >= 0:
        startValueJ = 0
    else:
        startValueJ = -x
    if x >= 0:
        endValueJ = W - x
    else:
        endValueJ = W

    for i in range(startValueI, endValueI):
        for j in range(startValueJ, endValueJ):
            result.append(colorArray[i][j])
    return result

def calculateSecondSelection(colorArray, x, y, coeff):
    result = []
    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // coeff
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    if y >= 0:
        startValueI = y
    else:
        startValueI = 0
    if y >= 0:
        endValueI = H
    else:
        endValueI = H + y
    if x >= 0:
        startValueJ = x
    else:
        startValueJ = 0
    if x >= 0:
        endValueJ = W
    else:
        endValueJ = W + x
    for i in range(startValueI, endValueI):
        for j in range(startValueJ, endValueJ):
            result.append(colorArray[i][j])
    return result


def calcAverage(array, imgSize):
    res = 0
    for digit in array:
        res += digit
    res /= len(array)
    return res

def calcDispers(array, imgSize):
    res = 0
    m = calcAverage(array, imgSize)
    for digit in array:
        res += (digit - m) ** 2
    res = res / (len(array) - 1)
    return res ** 0.5


def calcCorrelation(A, B, coeff):
    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // coeff
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    m1 = calcAverage(A, W * H)
    m2 = calcAverage(B, W * H)
    d1 = calcDispers(A, W * H)
    d2 = calcDispers(B, W * H)
    for i in range(len(A)):
        A[i] -= m1
        B[i] -= m2
        A[i] *= B[i]
    res = calcAverage(A, W * H) / (d1 * d2)
    return res



def autocorrelation(component, y, arrayRGB, color, coeff):
    print(f'in autocorrelation')
    array = []
    # color = []
    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // coeff
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    # for i in range(H):
    #     color2 = []
    #     for j in range(W):
    #         if (component == 2):
    #             color2.append(arrayRGB[i][j][2])
    #         if (component == 1):
    #             color2.append(arrayRGB[i][j][1])
    #         if (component == 0):
    #             color2.append(arrayRGB[i][j][0])
    #     color.append(color2)
    x = -199999
    for x in range(-W // 4, W // 4 + 1, 2):
        print(f'x in autcorel - {x}')
        firstSelection = calculateFirstSelection(color, x, y, coeff)
        secondSelection = calculateSecondSelection(color, x, y, coeff)
        array.append(calcCorrelation(firstSelection, secondSelection, coeff))
    return array


def bulding_graphics(rgb, YCbCr, coeff):
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    a = [i for i in range(-W // 4, W // 4 + 1, 2)]
    color_r = []
    color_g = []
    color_b = []


    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // coeff
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    for component in [0, 1, 2]:
        print(f'componet - {component}')
        for i in range(H):
            color2 = []
            for j in range(W):
                if (component == 2):
                    color2.append(rgb[i][j][2])
                if (component == 1):
                    color2.append(rgb[i][j][1])
                if (component == 0):
                    color2.append(rgb[i][j][0])

            if (component == 2):
                color_r.append(color2)
            if (component == 1):
                color_g.append(color2)
            if (component == 0):
                color_b.append(color2)

            # colors[component].append(color2)
    plt.title('Red')
    for i in range(-10, 11, 5):
        a2 = autocorrelation(2, i, rgb, color_r, coeff)
        plt.plot(a, a2)
    plt.legend(['-10', '-5', '0',  '5', '10'], loc=2)
    plt.show()
    plt.title('Green')
    for i in range(-10, 11, 5):
        a2 = autocorrelation(1, i, rgb, color_g, coeff)
        plt.plot(a, a2)
    plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    plt.show()
    plt.title('Blue')
    for i in range(-10, 11, 5):
        a2 = autocorrelation(0, i, rgb, color_b, coeff)
        plt.plot(a, a2)
    plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    plt.show()
    # plt.title('Cb')
    # for i in range(-10, 11, 5):
    #     a2 = autocorrelation(1, i, YCbCr)
    #     plt.plot(a, a2)
    # plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    # plt.show()
    # plt.title('Cr')
    # for i in range(-10, 11, 5):
    #     a2 = autocorrelation(2, i, YCbCr)
    #     plt.plot(a, a2)
    # plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    # plt.show()



def ToRGB(YCbCr):
    rgb = []
    for i in range((int.from_bytes(read_file.biHeight, byteorder='little'))):
        a = []
        for j in range((int.from_bytes(read_file.biWidth, byteorder='little'))):
            r = int(YCbCr[i][j][0] + 1.402 * (YCbCr[i][j][2] - 128))
            g = int(YCbCr[i][j][0] - 0.714 * (YCbCr[i][j][2] - 128) - 0.334 * (YCbCr[i][j][1]- 128))
            b = int(YCbCr[i][j][0] + 1.772 * (YCbCr[i][j][1] - 128))
            if r > 255:
                r = 255
            if g > 255:
                g = 255
            if b > 255:
                b = 255
            if r < 0:
                r = 0
            if b < 0:
                b = 0
            if g < 0:
                g = 0
            a.append([b, g, r])
        rgb.append(a)
    return rgb


def PSNR(pixel, rgb, component):
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    x1 = W * H * (2 ** 8 - 1) ** 2
    summ = 0
    for i in range(H):
        for j in range(W):
            summ += (pixel[i][j][component] - rgb[i][j][component]) ** 2
    return 10 * log10(x1 / summ)


def Decimation_excluding_even_rows_and_columns(img):
    # print(f'img - {img}')
    tmp = deepcopy(img)
    print(f'tmp- \n')
    for row in range(0, len(tmp[:6]), 1):
        for col in range(0, len(tmp[:6]), 1):
            print(f'row, col - {row, col}')
            print(f'- {tmp[row][col]}')
    # print(f'new tmp- \n')
    # for row in range(0, len(tmp[:11]), 1):
    #     for col in range(0, len(tmp[:11]), 1):
    #         print(f'- {tmp[row][col]}')
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(0, H, 2):
        for j in range(W):
            tmp[i][j][1] = None
            tmp[i][j][2] = None
    for i in range(H):
        for j in range(0, W, 2):
            tmp[i][j][1] = None
            tmp[i][j][2] = None

    decimated_cb_cr = []
    print(f'new tmp- \n')
    for row in range(0, len(tmp[:6])):
        for col in range(0, len(tmp[:6])):
            print(f'row, col - {row, col}')
            print(f'- {tmp[row][col]}')

    # Проходимся по каждой второй строке и столбцу и добавляем их значения в новую компоненту
    # for row in range(0, len(tmp), 2):
    #     decimated_row = []
    #     for col in range(0, len(tmp[0]), 2):
    #         decimated_row.append(tmp[row][col])
    #     decimated_cb_cr.append(decimated_row)

    return tmp


def Decimation_restoration(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(0, H, 2):
        for j in range(W):
            if i + 1 == H:
                tmp[i][j][1] = tmp[i - 1][j][1]
                tmp[i][j][2] = tmp[i - 1][j][2]
            if tmp[i][j][1] == None:
                tmp[i][j][1] = tmp[i + 1][j][1]
            if tmp[i][j][2] == None:
                tmp[i][j][2] = tmp[i + 1][j][2]
    for i in range(H):
        for j in range(0, W, 2):
            if j + 1 == W:
                tmp[i][j][1] = tmp[i][j - 1][1]
                tmp[i][j][2] = tmp[i][j - 1][2]
            if tmp[i][j][1] == None:
                tmp[i][j][1] = tmp[i][j + 1][1]
            if tmp[i][j][2] == None:
                tmp[i][j][2] = tmp[i][j + 1][2]
    return tmp


def Decimation_restoration2(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(0, H, 2):
        for j in range(W):
            if i + 1 == H:
                tmp[i][j][1] = tmp[i - 1][j][1]
                tmp[i][j][2] = tmp[i - 1][j][2]
            if tmp[i][j][1] == -1000:
                tmp[i][j][1] = tmp[i + 1][j][1]
            if tmp[i][j][2] == -1000:
                tmp[i][j][2] = tmp[i + 1][j][2]
    for i in range(H):
        for j in range(0, W, 2):
            if j + 1 == W:
                tmp[i][j][1] = tmp[i][j - 1][1]
                tmp[i][j][2] = tmp[i][j - 1][2]
            if tmp[i][j][1] == -1000:
                tmp[i][j][1] = tmp[i][j + 1][1]
            if tmp[i][j][2] == -1000:
                tmp[i][j][2] = tmp[i][j + 1][2]
    return tmp


def Decimation_excluding_even_rows_and_columns_method2(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(1, H, 2):
        for j in range(W):
            summ1 = (tmp[i][j][1] + tmp[i - 1][j][1]) // 2
            summ2 = (tmp[i][j][2] + tmp[i - 1][j][2]) // 2
            tmp[i][j][1] = summ1
            tmp[i][j][2] = summ2
            tmp[i - 1][j][1] = -1000
            tmp[i - 1][j][2] = -1000
    for i in range(H):
        for j in range(1, W, 2):
            summ1 = (tmp[i][j][1] + tmp[i][j - 1][1]) // 2
            summ2 = (tmp[i][j][2] + tmp[i][j - 1][2]) // 2
            tmp[i][j][1] = summ1
            tmp[i][j][2] = summ2
            tmp[i][j - 1][1] = -1000
            tmp[i][j - 1][2] = -1000
    return tmp

def Decimation_excluding_even_rows_and_columns_for_4(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(H - 3):
        for j in range(W):
            if i % 4 != 0:
                tmp[i][j][1] = -1000
                tmp[i][j][2] = -1000
    for i in range(H):
        for j in range(W - 3):
            if j % 4 != 0:
                tmp[i][j][1] = -1000
                tmp[i][j][2] = -1000
    return tmp


def Decimation_restoration_for_4(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(H - 3):
        for j in range(W):
            if i % 4 != 0:
                if i + 1 == H or i + 2 == H or i + 3 == H:
                    tmp[i][j][1] = tmp[i - (4 - i % 4)][j][1]
                    tmp[i][j][2] = tmp[i - (4 - i % 4)][j][2]
                if tmp[i][j][1] == -1000:
                    tmp[i][j][1] = tmp[i + (4 - i % 4)][j][1]
                if tmp[i][j][2] == -1000:
                    tmp[i][j][2] = tmp[i + (4 - i % 4)][j][2]
    for i in range(H):
        for j in range(W - 3):
            if j % 4 != 0:
                if j + 1 == W or j + 2 == W or j + 3 == W:
                    tmp[i][j][1] = tmp[i][j - (4 - j % 4)][1]
                    tmp[i][j][2] = tmp[i][j - (4 - j % 4)][2]
                if tmp[i][j][1] == -1000:
                    tmp[i][j][1] = tmp[i][j + (4 - j % 4)][1]
                if tmp[i][j][2] == -1000:
                    tmp[i][j][2] = tmp[i][j + (4 - j % 4)][2]
    return tmp

def Decimation_arithmetic_mean_for_4(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(0, H, 4):
        for j in range(W):
            summ1 = round((tmp[i][j][1] + tmp[i + 1][j][1] + tmp[i + 2][j][1] + tmp[i + 3][j][1]) / 4)
            summ2 = round((tmp[i][j][2] + tmp[i + 1][j][2] + tmp[i + 2][j][2] + tmp[i + 3][j][2]) / 4)
            tmp[i][j][1] = summ1
            tmp[i][j][2] = summ2
            tmp[i + 1][j][1] = -1000
            tmp[i + 1][j][2] = -1000
            tmp[i + 2][j][1] = -1000
            tmp[i + 2][j][2] = -1000
            tmp[i + 3][j][1] = -1000
            tmp[i + 3][j][2] = -1000
    for i in range(H):
        for j in range(0, W - 3, 4):
            if j % 4 != 0:
                summ1 = round((tmp[i][j][1] + tmp[i][j + 1][1] + tmp[i][j + 2][1] + tmp[i][j + 3][1]) / 4)
                summ2 = round((tmp[i][j][2] + tmp[i][j + 1][2] + tmp[i][j + 2][2] + tmp[i][j + 3][2]) / 4)
                tmp[i][j][1] = summ1
                tmp[i][j][2] = summ2
                tmp[i][j + 1][1] = -1000
                tmp[i][j + 2][2] = -1000
                tmp[i][j + 3][1] = -1000
                tmp[i][j + 1][2] = -1000
                tmp[i][j + 2][1] = -1000
                tmp[i][j + 3][2] = -1000
    return tmp


def Decimation_restoration_for_4_method_2(img):
    tmp = deepcopy(img)
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    for i in range(H):
        for j in range(W):
            if i % 4 != 0:
                if i + 1 == H or i + 2 == H or i + 3 == H:
                    tmp[i][j][1] = tmp[i - (4 - i % 4)][j][1]
                    tmp[i][j][2] = tmp[i - (4 - i % 4)][j][2]
                if tmp[i][j][1] == -1000:
                    tmp[i][j][1] = tmp[i + (4 - i % 4)][j][1]
                if tmp[i][j][2] == -1000:
                    tmp[i][j][2] = tmp[i + (4 - i % 4)][j][2]
    for i in range(H):
        for j in range(W):
            if j % 4 != 0:
                if j + 1 == W or j + 2 == W or j + 3 == W:
                    tmp[i][j][1] = tmp[i][j - (4 - j % 4)][1]
                    tmp[i][j][2] = tmp[i][j - (4 - j % 4)][2]
                if tmp[i][j][1] == -1000:
                    tmp[i][j][1] = tmp[i][j + (4 - j % 4)][1]
                if tmp[i][j][2] == -1000:
                    tmp[i][j][2] = tmp[i][j + (4 - j % 4)][2]
    return tmp


def count_component(component, pixels, num):  # num - 0, значит RGB, num - 1, значит YCbCr
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    frequency = [0] * 256
    values = [i for i in range(256)]
    for i in range(H):
        for j in range(W):
            frequency[pixels[i][j][component]] += 1
    if component == 0 and num == 0:
        plt.title('Гистограмма B')
    elif component == 1 and num == 0:
        plt.title('Гистограмма G')
    elif component == 2 and num == 0:
        plt.title('Гистограмма R')
    elif component == 0 and num == 1:
        plt.title('Гистограмма Y')
    elif component == 1 and num == 1:
        plt.title('Гистограмма Cb')
    else:
        plt.title('Гистограмма Cr')
    plt.bar(values, frequency)
    plt.show()
    return values

def entropy(component, pixels, num): #num - 0, значит RGB, num - 1, значит YCbCr
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    frequency = [0] * 256
    for i in range(H):
        for j in range(W):
            frequency[pixels[i][j][component]] += 1
    summ = 0
    for x in range(256):
        p = frequency[x] / sum(frequency)
        if p != 0:
            summ += p * log2(p)
    H = -summ
    if component == 0 and num == 0:
        print(f"Энтропия B = {H}")
    elif component == 1 and num == 0:
        print(f"Энтропия G = {H}")
    elif component == 2 and num == 0:
        print(f"Энтропия R = {H}")
    elif component == 0 and num == 1:
        print(f"Энтропия Y = {H}")
    elif component == 1 and num == 1:
        print(f"Энтропия Cb = {H}")
    else:
        print(f"Энтропия Cr = {H}")


def neighbor_left(pixels, component):
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    d = []
    for i in range(1, H):
        for j in range(1, W):
            d.append(pixels[i][j][component] - pixels[i][j - 1][component])
    return d


def neighbor_above(pixels, component):
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    d = []
    for i in range(1, H):
        for j in range(W):
            d.append(pixels[i][j][component] - pixels[i - 1][j][component])
    return d


def neighbor_left_above(pixels, component):
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    d = []
    for i in range(1, H):
        for j in range(1, W):
            d.append(pixels[i][j][component] - pixels[i - 1][j - 1][component])
    return d


def neighbor_average_left_above(pixels, component):
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    d = []
    for i in range(1, H):
        for j in range(1, W):
            average = (pixels[i - 1][j][component] + pixels[i][j - 1][component] + pixels[i - 1][j - 1][component]) // 3
            d.append(pixels[i][j][component] - average)
    return d


def count(array):
    frequency1 = [0] * 256
    frequency2 = [0] * 255
    for i in range(len(array)):
        if array[i] >= 0:
            frequency1[array[i]] += 1
        else:
            frequency2[abs(array[i]) - 1] += 1
    frequency2.reverse()
    frequency = frequency2 + frequency1
    return frequency

def building_histograms(pixels, YCbCr):
    dR = [neighbor_left(pixels, 2), neighbor_above(pixels, 2), neighbor_left_above(pixels, 2), neighbor_average_left_above(pixels, 2)]
    dG = [neighbor_left(pixels, 1), neighbor_above(pixels, 1), neighbor_left_above(pixels, 1), neighbor_average_left_above(pixels, 1)]
    dB = [neighbor_left(pixels, 0), neighbor_above(pixels, 0), neighbor_left_above(pixels, 0), neighbor_average_left_above(pixels, 0)]
    dY = [neighbor_left(YCbCr, 0), neighbor_above(YCbCr, 0), neighbor_left_above(YCbCr, 0), neighbor_average_left_above(YCbCr, 0)]
    dCb = [neighbor_left(YCbCr, 1), neighbor_above(YCbCr, 1), neighbor_left_above(YCbCr, 1), neighbor_average_left_above(YCbCr, 1)]
    dCr = [neighbor_left(YCbCr, 2), neighbor_above(YCbCr, 2), neighbor_left_above(YCbCr, 2), neighbor_average_left_above(YCbCr, 2)]
    values = [i for i in range(-255, 256)]
    plt.subplot(221)
    plt.title("R правило 1")
    plt.bar(values, count(dR[0]))
    plt.ylim([0, 40000])

    plt.subplot(222)
    plt.title("R правило 2")
    plt.bar(values, count(dR[1]))
    plt.ylim([0, 40000])
    # plt.show()

    plt.subplot(223)
    plt.title("R правило 3")
    plt.bar(values, count(dR[2]))
    plt.ylim([0, 40000])

    plt.subplot(224)
    plt.title("R правило 4")
    plt.bar(values, count(dR[3]))
    plt.ylim([0, 40000])
    plt.show()

    plt.subplot(221)
    plt.title("G правило 1")
    plt.bar(values, count(dG[0]))
    plt.ylim([0, 40000])

    plt.subplot(222)
    plt.title("G правило 2")
    plt.bar(values, count(dG[1]))
    plt.ylim([0, 40000])

    # plt.show()
    plt.subplot(223)
    plt.title("G правило 3")
    plt.bar(values, count(dG[2]))
    plt.ylim([0, 40000])

    plt.subplot(224)
    plt.title("G правило 4")
    plt.bar(values, count(dG[3]))
    plt.ylim([0, 40000])
    plt.show()

    plt.subplot(221)
    plt.title("B правило 1")
    plt.bar(values, count(dB[0]))
    plt.ylim([0, 40000])

    plt.subplot(222)
    plt.title("B правило 2")
    plt.bar(values, count(dB[1]))
    plt.ylim([0, 40000])
    # plt.show()

    plt.subplot(223)
    plt.title("B правило 3")
    plt.bar(values, count(dB[2]))
    plt.ylim([0, 40000])

    plt.subplot(224)
    plt.title("B правило 4")
    plt.bar(values, count(dB[3]))
    plt.ylim([0, 40000])
    plt.show()

    plt.subplot(221)
    plt.title("Y правило 1")
    plt.bar(values, count(dY[0]))
    plt.ylim([0, 40000])

    plt.subplot(222)
    plt.title("Y правило 2")
    plt.bar(values, count(dY[1]))
    plt.ylim([0, 40000])

    # plt.show()
    plt.subplot(223)
    plt.title("Y правило 3")
    plt.bar(values, count(dY[2]))
    plt.ylim([0, 40000])

    plt.subplot(224)
    plt.title("Y правило 4")
    plt.bar(values, count(dY[3]))
    plt.ylim([0, 40000])
    plt.show()

    plt.subplot(221)
    plt.title("Cb правило 1")
    plt.bar(values, count(dCb[0]))
    plt.ylim([0, 40000])

    plt.subplot(222)
    plt.title("Cb правило 2")
    plt.bar(values, count(dCb[1]))
    plt.ylim([0, 40000])

    # plt.show()
    plt.subplot(223)
    plt.title("Cb правило 3")
    plt.bar(values, count(dCb[2]))
    plt.ylim([0, 40000])

    plt.subplot(224)
    plt.title("Cb правило 4")
    plt.bar(values, count(dCb[3]))
    plt.ylim([0, 40000])
    plt.show()

    plt.subplot(221)
    plt.title("Cr правило 1")
    plt.bar(values, count(dCr[0]))
    plt.ylim([0, 40000])

    plt.subplot(222)
    plt.title("Cr правило 2")
    plt.bar(values, count(dCr[1]))
    plt.ylim([0, 40000])

    # plt.show()
    plt.subplot(223)
    plt.title("Cr правило 3")
    plt.bar(values, count(dCr[2]))
    plt.ylim([0, 40000])

    plt.subplot(224)
    plt.title("Cb правило 4")
    plt.bar(values, count(dCr[3]))
    plt.ylim([0, 40000])
    plt.show()


def entropy_rules(pixels, YCbCr):
    dR = [neighbor_left(pixels, 2), neighbor_above(pixels, 2), neighbor_left_above(pixels, 2), neighbor_average_left_above(pixels, 2)]
    dG = [neighbor_left(pixels, 1), neighbor_above(pixels, 1), neighbor_left_above(pixels, 1), neighbor_average_left_above(pixels, 1)]
    dB = [neighbor_left(pixels, 0), neighbor_above(pixels, 0), neighbor_left_above(pixels, 0), neighbor_average_left_above(pixels, 0)]
    dY = [neighbor_left(YCbCr, 0), neighbor_above(YCbCr, 0), neighbor_left_above(YCbCr, 0), neighbor_average_left_above(YCbCr, 0)]
    dCb = [neighbor_left(YCbCr, 1), neighbor_above(YCbCr, 1), neighbor_left_above(YCbCr, 1), neighbor_average_left_above(YCbCr, 1)]
    dCr = [neighbor_left(YCbCr, 2), neighbor_above(YCbCr, 2), neighbor_left_above(YCbCr, 2), neighbor_average_left_above(YCbCr, 2)]
    print(f'Энтропия для массивов разности:\n')
    for i in range(len(dR)):
        frequency = count(dR[i])
        summ = 0
        for x in range(len(frequency)):
            p = frequency[x] / sum(frequency)
            if p != 0:
                summ += p * log2(p)
        H = -summ
        print(f'R{i + 1} = {H}')
    for i in range(len(dG)):
        frequency = count(dG[i])
        summ = 0
        for x in range(len(frequency)):
            p = frequency[x] / sum(frequency)
            if p != 0:
                summ += p * log2(p)
        H = -summ
        print(f'G{i + 1} = {H}')
    for i in range(len(dB)):
        frequency = count(dB[i])
        summ = 0
        for x in range(len(frequency)):
            p = frequency[x] / sum(frequency)
            if p != 0:
                summ += p * log2(p)
        H = -summ
        print(f'B{i + 1} = {H}')
    for i in range(len(dY)):
        frequency = count(dY[i])
        summ = 0
        for x in range(len(frequency)):
            p = frequency[x] / sum(frequency)
            if p != 0:
                summ += p * log2(p)
        H = -summ
        print(f'Y{i + 1} = {H}')
    for i in range(len(dCb)):
        frequency = count(dCb[i])
        summ = 0
        for x in range(len(frequency)):
            p = frequency[x] / sum(frequency)
            if p != 0:
                summ += p * log2(p)
        H = -summ
        print(f'Cb{i + 1} = {H}')
    for i in range(len(dCr)):
        frequency = count(dCr[i])
        summ = 0
        for x in range(len(frequency)):
            p = frequency[x] / sum(frequency)
            if p != 0:
                summ += p * log2(p)
        H = -summ
        print(f'Cr{i + 1} = {H}')


def dop_get_four_subframe(YCbCr):
    tmp1 = YCbCr

    tmp2 = YCbCr
    tmp3 = YCbCr
    tmp4 = YCbCr
    print(f'tmp1- \n')

    # for row in range(0, len(tmp[:6]), 1):
    #     for col in range(0, len(tmp[:6]), 1):
    #         print(f'row, col - {row, col}')
    #         print(f'- {tmp[row][col]}')
    # print(f'new tmp- \n')
    # for row in range(0, len(tmp[:11]), 1):
    #     for col in range(0, len(tmp[:11]), 1):
    #         print(f'- {tmp[row][col]}')
    H = int.from_bytes(read_file.biHeight, byteorder='little')
    W = int.from_bytes(read_file.biWidth, byteorder='little')
    # 2y + i; 2x + j
    counterH = 0
    counterW = 0
    counter = 0
    # i, j = 0, 0
    x, y = W/2, H/2

    for row in range(0, len(tmp1[:6]), 1):
        for col in range(0, len(tmp1[:6]), 1):
            print(f'row, col - {row, col}')
            print(f'- {tmp1[row][col]}')
    print(f'tmp1- \n')

    tmp1_new = [[0 for j in range(int(x))] for i in range(int(y))]
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            counter += 1
            tmp1_new[int(i / 2)][int(j / 2)] = tmp1[i][j]

    print(f'x, y {x, y}, counter = {counter}')

    for row in range(0, len(tmp1_new[:6]), 1):
        for col in range(0, len(tmp1_new[:6]), 1):
            print(f'row, col - {row, col}')
            print(f'- {tmp1_new[row][col]}')
    print(f'new tmp- \n')
            # tmp1_new[int(i/2)][int(j/2)] = tmp1[i][j]

    x, y = W / 2, H / 2
    tmp2_new = [[0 for j in range(int(x))] for i in range(int(y))]
    for i in range(0, H, 2):
        for j in range(1, W, 2):
            tmp2_new[int(i/2)][int(j/2)] = tmp2[i][j]

    x, y = W / 2, H / 2
    tmp3_new = [[0 for j in range(int(x))] for i in range(int(y))]
    for i in range(1, H, 2):
        for j in range(0, W, 2):
            tmp3_new[int(i / 2)][int(j / 2)] = tmp3[i][j]

    x, y = W / 2, H / 2
    tmp4_new = [[0 for j in range(int(x))] for i in range(int(y))]
    for i in range(1, H, 2):
        for j in range(1, W, 2):
            tmp4_new[int(i / 2)][int(j / 2)] = tmp4[i][j]

    return tmp1_new, tmp2_new, tmp3_new, tmp4_new

    #
    # for row in range(0, len(tmp1[:6]), 1):
    #     for col in range(0, len(tmp1[:6]), 1):
    #         print(f'row, col - {row, col}')
    #         print(f'- {tmp1[row][col]}')
    # print(f'new tmp- \n')
    #
    # print(f'H = {H}; W = {W}; counrerH = {counterH}; counterW = {counterW}')
    #
    # # mask = np.where(tmp1 != -1000, True, False)
    # # tmp1_result = tmp1[mask]
    #
    # counterH = 0
    # counterW = 0
    # # i, j = 0, 1
    # for i in range(1, H, 2):
    #     for j in range(W):
    #         tmp2[i][j][0] = -1000
    #         counterH += 1
    # for i in range(H):
    #     for j in range(2, W):
    #         if j % 3 != 2:
    #             tmp2[i][j][0] = -1000
    #             counterW += 1
    #
    # print(f'H = {H}; W = {W}; counrerH = {counterH}; counterW = {counterW}')
    # counterH = 0
    # counterW = 0
    # # i, j = 1, 0
    # for i in range(2, H):
    #     for j in range(W):
    #         if i % 3 != 2:
    #             tmp3[i][j][0] = -1000
    #             counterH += 1
    # for i in range(H):
    #     for j in range(1, W, 2):
    #         tmp3[i][j][0] = -1000
    #         counterW += 1
    # print(f'H = {H}; W = {W}; counrerH = {counterH}; counterW = {counterW}')
    # counterH = 0
    # counterW = 0
    # # i, j = 1, 1
    # for i in range(2, H):
    #     for j in range(W):
    #         if i % 3 != 2:
    #             tmp4[i][j][0] = -1000
    #             counterH += 1
    # for i in range(H):
    #     for j in range(2, W):
    #         if i % 3 != 2:
    #             tmp4[i][j][0] = -1000
    #             counterW += 1
    #
    # print(f'H = {H}; W = {W}; counrerH = {counterH}; counterW = {counterW}')
    #
    #
    # for i in range(int(H/2)):
    #     for j in range(int(W/2)):
    #         if tmp1[i][j][0] != -1000:
    #             del tmp1[i][j]
    #
    # for i in range(int(H / 2)):
    #     for j in range(int(W / 3)):
    #         if tmp2[i][j][0] != -1000:
    #             del tmp2[i][j]
    #
    #
    #         if tmp3[i][j][0] == -1000:
    #             del tmp3[i][j]
    #         if tmp4[i][j][0] == -1000:
    #             del tmp4[i][j]
    #
    # for row in range(0, len(tmp1[:6]), 1):
    #     for col in range(0, len(tmp1[:6]), 1):
    #         print(f'row, col - {row, col}')
    #         print(f'- {tmp1[row][col]}')
    # print(f'new tmp- \n')
    # i, j = 0, 0

    # for i in range(1, H-1, 2):
    #     for j in range(W):
    #         if tmp1[i][j][0] == -1000:
    #             tmp1[i][j][0] = tmp1[i+1][j][0]
    # for i in range(H):
    #     for j in range(1, W-1, 2):
    #         if tmp1[i][j][0] == -1000:
    #             tmp1[i][j][0] = tmp1[i][j + 1][0]
    #         tmp1[i][j][0] = -1000
    #
    # # i, j = 0, 1
    # for i in range(1, H-1, 2):
    #     for j in range(W):
    #         if tmp2[i][j][0] == -1000:
    #             tmp2[i][j][0] = tmp1[i + 1][j][0]
    #
    # is_first = True
    # for i in range(H):
    #     for j in range(2, W):
    #         if tmp2[i][j][0] == -1000:
    #             tmp2[i][j][0] = tmp2[i][j + 7][0]
    #             # if is_first:
    #             #     tmp2[i][j][0] = tmp2[i][j+3][0] # -1000 -1000 5 -1000 -1000 5 -1000 -1000 5 -1000 -1000 5 -1000 -1000 6
    #         if j % 3 != 2:
    #             tmp2[i][j][0] = -1000



                # decimated_cb_cr = []
    # print(f'new tmp- \n')
    # for row in range(0, len(tmp[:6])):
    #     for col in range(0, len(tmp[:6])):
    #         print(f'row, col - {row, col}')
    #         print(f'- {tmp[row][col]}')



def dop_write_four_subframe(subframe1, subframe2, subframe3, subframe4):
    f1 = open(f"img/{read_file.count} изображение.bmp", 'wb')
    H = int(int.from_bytes(read_file.biHeight, byteorder='little') / 2)
    W = int(int.from_bytes(read_file.biWidth, byteorder='little') / 2)
    read_file.count += 1

    f1.write(read_file.bfType)
    f1.write(read_file.bfSize)
    f1.write(read_file.bfReserved)
    f1.write(read_file.bfOffBits)
    f1.write(read_file.biSize)
    f1.write(read_file.biWidth)
    f1.write(read_file.biHeight)
    f1.write(read_file.biPlanes)
    f1.write(read_file.biBitCount)
    f1.write(read_file.biCompression)
    f1.write(read_file.biSizeImage)
    f1.write(read_file.biXPelsPerMeter)
    f1.write(read_file.biYPelsPerMeter)
    f1.write(read_file.biClrUsed)
    f1.write(read_file.biCrlImportant)


    for i in range(H):
        for j in range(W):
            # изменить заголовок
            Y = subframe1[i][j][0]
            f1.write(bytes([Y]))
            f1.write(bytes([Y]))
            f1.write(bytes([Y]))
    f1.write(read_file.ost)
    f1.close()

    f2 = open(f"img/{read_file.count} изображение.bmp", 'wb')
    read_file.count += 1

    f2.write(read_file.bfType)
    f2.write(read_file.bfSize)
    f2.write(read_file.bfReserved)
    f2.write(read_file.bfOffBits)
    f2.write(read_file.biSize)
    f2.write(read_file.biWidth)
    f2.write(read_file.biHeight)
    f2.write(read_file.biPlanes)
    f2.write(read_file.biBitCount)
    f2.write(read_file.biCompression)
    f2.write(read_file.biSizeImage)
    f2.write(read_file.biXPelsPerMeter)
    f2.write(read_file.biYPelsPerMeter)
    f2.write(read_file.biClrUsed)
    f2.write(read_file.biCrlImportant)


    for i in range(H):
        for j in range(
                W):
            Y = subframe2[i][j][0]
            f2.write(bytes([Y]))
            f2.write(bytes([Y]))
            f2.write(bytes([Y]))
    f2.write(read_file.ost)
    f2.close()

    f3 = open(f"img/{read_file.count} изображение.bmp", 'wb')
    read_file.count += 1

    f3.write(read_file.bfType)
    f3.write(read_file.bfSize)
    f3.write(read_file.bfReserved)
    f3.write(read_file.bfOffBits)
    f3.write(read_file.biSize)
    f3.write(read_file.biWidth)
    f3.write(read_file.biHeight)
    f3.write(read_file.biPlanes)
    f3.write(read_file.biBitCount)
    f3.write(read_file.biCompression)
    f3.write(read_file.biSizeImage)
    f3.write(read_file.biXPelsPerMeter)
    f3.write(read_file.biYPelsPerMeter)
    f3.write(read_file.biClrUsed)
    f3.write(read_file.biCrlImportant)

    for i in range(H):
        for j in range(
                W):
            Y = subframe3[i][j][0]
            f3.write(bytes([Y]))
            f3.write(bytes([Y]))
            f3.write(bytes([Y]))
    f3.write(read_file.ost)
    f3.close()

    f4 = open(f"img/{read_file.count} изображение.bmp", 'wb')
    read_file.count += 1

    f4.write(read_file.bfType)
    f4.write(read_file.bfSize)
    f4.write(read_file.bfReserved)
    f4.write(read_file.bfOffBits)
    f4.write(read_file.biSize)
    f4.write(read_file.biWidth)
    f4.write(read_file.biHeight)
    f4.write(read_file.biPlanes)
    f4.write(read_file.biBitCount)
    f4.write(read_file.biCompression)
    f4.write(read_file.biSizeImage)
    f4.write(read_file.biXPelsPerMeter)
    f4.write(read_file.biYPelsPerMeter)
    f4.write(read_file.biClrUsed)
    f4.write(read_file.biCrlImportant)

    for i in range(H):
        for j in range(
                W):
            Y = subframe4[i][j][0]
            f4.write(bytes([Y]))
            f4.write(bytes([Y]))
            f4.write(bytes([Y]))
    f4.write(read_file.ost)
    f4.close()


def create_bmp2(width, height, pixels, YCbCr, filename):
    # Заголовок BMP файла
    f1 = open(filename, 'wb')

    f1.write(read_file.bfType)
    f1.write(struct.pack('<I', len(pixels)))
    f1.write(read_file.bfReserved)
    f1.write(read_file.bfOffBits)
    f1.write(read_file.biSize)
    f1.write(struct.pack('<I', width))
    f1.write(struct.pack('<I', height))
    f1.write(read_file.biPlanes)
    f1.write(read_file.biBitCount)
    f1.write(read_file.biCompression)
    f1.write(struct.pack('<I', len(pixels)))
    f1.write(read_file.biXPelsPerMeter)
    f1.write(read_file.biYPelsPerMeter)
    f1.write(read_file.biClrUsed)
    f1.write(read_file.biCrlImportant)

    byte_array = bytearray()
    сounter_row = 0
    counter_col = 0
    # for row in pixels:
    #     сounter_row += 1
    #     for pixel in row:
    #         counter_col += 1
            # Y = pixels[i][j][0]
            # f1.write(pixel)
            # f1.write(bytes(pixel[0]))
            # f1.write(bytes(pixel[1]))
            # f1.write(bytes(pixel[2]))
    f1.write(pixels)
    f1.write(read_file.ost)
    f1.close()
            # Y = pixel[0]
            # # Преобразование значений [R, G, B] в формат байтов и добавление их в массив
            # pixel[1] = Y
            # pixel[2] = Y
            # byte_array.extend(bytes(pixel))
    print(f'сounter_row - {сounter_row}, counter_col - {counter_col}')




def create_bmp(width, height, pixels, filename):
    # Заголовок BMP файла
    header = bytearray(b'BM')  # Сигнатура файла
    header += struct.pack('<I', len(pixels))  # Размер файла
    header += bytearray(4)  # Зарезервированные байты
    header += struct.pack('<I', 54)  # Смещение данных
    header += struct.pack('<I', 40)  # Размер заголовка
    header += struct.pack('<I', width)  # Ширина изображения
    header += struct.pack('<I', height)  # Высота изображения
    header += struct.pack('<H', 1)  # Число плоскостей
    header += struct.pack('<H', 24)  # Бит на пиксель
    header += struct.pack('<I', 0)  # Сжатие
    header += struct.pack('<I', len(pixels))  # Размер данных изображения
    header += struct.pack('<I', 0)  # X пикселей на метр
    header += struct.pack('<I', 0)  # Y пикселей на метр
    header += struct.pack('<I', 0)  # Количество используемых цветов
    header += struct.pack('<I', 0)  # Важные цвета
    # Записываем данные в файл
    with open(filename, 'wb') as f:
        f.write(header + pixels)

def convert_pixels_to_bytes(pixels):
    byte_array = bytearray()
    сounter_row = 0
    counter_col = 0
    for row in pixels:
        сounter_row += 1
        for pixel in row:
            counter_col += 1
            Y = pixel[0]
            # # Преобразование значений [R, G, B] в формат байтов и добавление их в массив
            pixel[1] = Y
            pixel[2] = Y
            byte_array.extend(bytes(pixel))
    print(f'сounter_row - {сounter_row}, counter_col - {counter_col}')
    return byte_array

def dop_write_one_file2(YCbCr, coeff, index):
    filename = f"img/{index} изображение.bmp"
    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // coeff
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // coeff
    byte_array = convert_pixels_to_bytes(YCbCr)
    # create_bmp(W, H, byte_array, filename)
    create_bmp2(W, H, byte_array,YCbCr, filename)


def dop_write_one_file(YCbCr, coeff, index):
    # f = open(f"D://Папка//Децимация//перевод YCbCr после децимация в {coeff} раза 1 способом.bmp", 'wb')
    f = open(f"img/{index} изображение.bmp", 'wb')
    H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    f.write(read_file.bfType)
    f.write(read_file.bfSize)
    f.write(read_file.bfReserved)
    f.write(read_file.bfOffBits)
    f.write(read_file.biSize)
    f.write(read_file.biWidth)
    f.write(read_file.biHeight)
    f.write(read_file.biPlanes)
    f.write(read_file.biBitCount)
    f.write(read_file.biCompression)
    f.write(read_file.biSizeImage)
    f.write(read_file.biXPelsPerMeter)
    f.write(read_file.biYPelsPerMeter)
    f.write(read_file.biClrUsed)
    f.write(read_file.biCrlImportant)

    print(f'H - {H}; H // coeff - {H // coeff} W - {W} W // coeff - {W // coeff}; YCbCr shape - {np.array(YCbCr).shape}')
    YCbCr2 = []
    for i in range(H // coeff):
        a = []
        for j in range(W // coeff):
            Y = YCbCr[i][j][0]
            Cb = YCbCr[i][j][1]
            Cr = YCbCr[i][j][2]
            f.write(bytes([Y]))
            f.write(bytes([Cb]))
            f.write(bytes([Cr]))
            a.append([Y, Cb, Cr])

            # for j in range(W // coeff):
            #     for z in range(coeff):
            #         Y = YCbCr[i][j][0]
            #         Cb = YCbCr[i][j][1]
            #         Cr = YCbCr[i][j][2]
            #         f.write(bytes([Y]))
            #         f.write(bytes([Cb]))
            #         f.write(bytes([Cr]))
            #         a.append([Y, Cb, Cr])
        YCbCr2.append(a)
    f.write(read_file.ost)
    f.close()
    return YCbCr2


def dop_autocorrelation(YCbCr, number):
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // 2
    a = [i for i in range(-W // 4, W // 4 + 1, 2)]
    color_r = []
    color_g = []
    color_b = []
    color_Y = []
    color_Cb = []
    color_Cr = []


    H = (int.from_bytes(read_file.biHeight, byteorder='little')) // 2
    W = (int.from_bytes(read_file.biWidth, byteorder='little')) // 2

    for i in range(H):
        color2 = []
        for j in range(W):
           color2.append(YCbCr[i][j][0])
        color_Y.append(color2)

    plt.title(f'Y {number}')
    for i in range(-10, 11, 5):
        a2 = autocorrelation(2, i, YCbCr, color_Y, 2)
        plt.plot(a, a2)
    plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    plt.show()

    # for component in [0, 1, 2]:
    #     print(f'componet - {component}')
    #     for i in range(H):
    #         color2 = []
    #         for j in range(W):
    #             if (component == 2):
    #                 color2.append(rgb[i][j][2])
    #             if (component == 1):
    #                 color2.append(rgb[i][j][1])
    #             if (component == 0):
    #                 color2.append(rgb[i][j][0])
    #
    #         if (component == 2):
    #             color_r.append(color2)
    #         if (component == 1):
    #             color_g.append(color2)
    #         if (component == 0):
    #             color_b.append(color2)

            # colors[component].append(color2)

    # plt.title('Cb')
    # for i in range(-10, 11, 5):
    #     a2 = autocorrelation(1, i, YCbCr)
    #     plt.plot(a, a2)
    # plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    # plt.show()
    # plt.title('Cr')
    # for i in range(-10, 11, 5):
    #     a2 = autocorrelation(2, i, YCbCr)
    #     plt.plot(a, a2)
    # plt.legend(['-10', '-5', '0', '5', '10'], loc=2)
    # plt.show()



def main():
    filename = 'img/img.bmp'
    readen_file = read_file()
    readen_file.print_fields()
    pixel = readen_file.pixels
    # print(pixel)
    writeFile(pixel)
    YCbCr = ToYCbCr(pixel)
    writeFile(YCbCr)

    print('--------------------')
    print(
        f'Корреляция RGB: \nRG - {CorelCoeff(pixel, 1, 2)}, RB - {CorelCoeff(pixel, 2, 0)}, GB - {CorelCoeff(pixel, 1, 0)}')  # 4а)
    print(
        f'Корреляция YCbCr: \nYCb - {CorelCoeff(YCbCr, 0, 1)}, YCr - {CorelCoeff(YCbCr, 0, 2)}, CbCr - {CorelCoeff(YCbCr, 1, 2)}')  # 5
    # calc_autocorel_coeff()

    # bulding_graphics(pixel, YCbCr, 1)  # 4b


    rgb = ToRGB(YCbCr)  # перевод YCbCr в RGB
    write_file(rgb)
    print("PSNR синий - ",PSNR(pixel, rgb, 0), "PSNR зеленый - ",PSNR(pixel, rgb, 1),"PSNR красный - ",PSNR( pixel, rgb, 2)) #PSNR для всего (задание 7)

    # Задания 8 - 11
    YCbCr2_with_Decimation_method1 = Decimation_excluding_even_rows_and_columns(YCbCr)
    YCbCr2_with_Decimation_method2 = Decimation_excluding_even_rows_and_columns_method2(
        YCbCr)  # с 2 способом децимация в два раза
    # YCbCr2_with_Decimation2 = Decimation_excluding_even_rows_and_columns2(YCbCr)# с 1 способом децимация в два раза
    YCbCr2_after_Detimination_method1 = Decimation_restoration(YCbCr2_with_Decimation_method1)
    YCbCr2_after_Detimination_method2 = Decimation_restoration2(
        YCbCr2_with_Decimation_method2)
    # H = int.from_bytes(read_file.biHeight, byteorder='little')
    # W = int.from_bytes(read_file.biWidth, byteorder='little')
    # ycbcrrestored = restore_cb_cr(YCbCr2_with_Decimation2,W,H)
    RGB_after_Detimination_method1 = ToRGB(YCbCr2_after_Detimination_method1)
    RGB_after_Detimination_method2 = ToRGB(YCbCr2_after_Detimination_method2)
    # RGB22 = ToRGB(ycbcrrestored)
    write_file(RGB_after_Detimination_method1)
    write_file(RGB_after_Detimination_method2)
    # write_file(RGB22)

    # PSNR для обоих способов при уменьшении в два раза - 9 задание

    print(
        f"PSNR после 1 способа при уменьшении в 2 раза\n Синий = {PSNR(pixel, RGB_after_Detimination_method1, 0)}, "
        f"Зеленый = {PSNR(pixel, RGB_after_Detimination_method1, 1)}, Красный = {PSNR(pixel, RGB_after_Detimination_method1, 2)},"
        f"Cb = {PSNR(YCbCr, YCbCr2_after_Detimination_method1, 1)}, Cr = {PSNR(YCbCr, YCbCr2_after_Detimination_method1, 2)}")
    print(
        f"PSNR после 2 способа при уменьшении в 2 раза\n Синий = {PSNR(pixel, RGB_after_Detimination_method2, 0)}, "
        f"Зеленый = {PSNR(pixel, RGB_after_Detimination_method2, 1)}, Красный = {PSNR(pixel, RGB_after_Detimination_method2, 2)},"
        f"Cb = {PSNR(YCbCr, YCbCr2_after_Detimination_method2, 1)}, Cr = {PSNR(YCbCr, YCbCr2_after_Detimination_method2, 2)}")

    YCbCr4_with_Decimation_method1 = Decimation_excluding_even_rows_and_columns_for_4(
        YCbCr)  # с 1 способом децимация в четыре раза
    YCbCr4_after_Detimination_method1 = Decimation_restoration_for_4(
        YCbCr4_with_Decimation_method1)  # с 1 способом восстановление YCbCr после децимации

    RGB_after_Detimination_method1 = ToRGB(
        YCbCr4_after_Detimination_method1)  # с 1 способом в 4 раза перевод обратно в RGB
    write_file(
        RGB_after_Detimination_method1)  # восстановление - а) 9 изображение

    YCbCr4_with_Decimation_method2 = Decimation_arithmetic_mean_for_4(
        YCbCr)  # с 2 способом децимация в 4 раза

    YCbCr4_after_Detimination_method2 = Decimation_restoration_for_4_method_2(
        YCbCr4_with_Decimation_method2)  # восстановление после 2 способа

    RGB4_after_Detimination_method2 = ToRGB(
        YCbCr4_after_Detimination_method2)
    write_file(
        RGB4_after_Detimination_method2)  # восстановление после 2 способа - b) 10 изображение

    print(
        f"PSNR после 1 способа при уменьшении в 4 раза\n Синий = {PSNR(pixel, RGB_after_Detimination_method1, 0)}, "
        f"Зеленый = {PSNR(pixel, RGB_after_Detimination_method1, 1)}, Красный = {PSNR(pixel, RGB_after_Detimination_method1, 2)},"
        f"Cb = {PSNR(YCbCr, YCbCr4_after_Detimination_method1, 1)}, Cr = {PSNR(YCbCr, YCbCr4_after_Detimination_method1, 2)}")
    print(
        f"PSNR после 2 способа при уменьшении в 4 раза\n Синий = {PSNR(pixel, RGB4_after_Detimination_method2, 0)}, "
        f"Зеленый = {PSNR(pixel, RGB4_after_Detimination_method2, 1)}, Красный = {PSNR(pixel, RGB4_after_Detimination_method2, 2)},"
        f"Cb = {PSNR(YCbCr, YCbCr4_after_Detimination_method2, 1)}, Cr = {PSNR(YCbCr, YCbCr4_after_Detimination_method2, 2)}")

    # print(pixel)
    for x in range(3): #Вывод гистограмм - задание 12
      count_component(x, pixel, 0)
      count_component(x, YCbCr, 1)

    for x in range(3):  # Энтропия - задание 13
        entropy(x, pixel, 0)
        entropy(x, YCbCr, 1)

    # print(
    #     f"Правило сосед слева: \nB - {neighbor_left(pixel, 0)} - {len(neighbor_left(pixel, 0))}")  # Пример вывода массива

    building_histograms(pixel,
                        YCbCr)  # задание 15, построение гистограмм по правилам (верно???)

    entropy_rules(pixel, YCbCr)  # задание 16, энтропия

    # subframe1, subframe2, subframe3, subframe4 = dop_get_four_subframe(YCbCr)
    # dop_write_four_subframe(subframe1, subframe2, subframe3, subframe4)
    # # subframe1_help = dop_write_one_file(YCbCr, 1, 1)
    # H = (int.from_bytes(read_file.biHeight, byteorder='little'))
    # W = (int.from_bytes(read_file.biWidth, byteorder='little'))
    # dop_hight = H // 2
    # dop_wight = W // 2
    # dop_write_one_file2(subframe1, 2, 1)
    # dop_write_one_file2(subframe2, 2, 2)
    # dop_write_one_file2(subframe3, 2, 3)
    # dop_write_one_file2(subframe4, 2, 4)
    # dop_autocorrelation(YCbCr, 0)
    #
    # dop_autocorrelation(subframe1, 1)
    # dop_autocorrelation(subframe2, 2)
    # dop_autocorrelation(subframe3, 3)
    # dop_autocorrelation(subframe4, 4)

    # print(
    #     f"Правило сосед слева: \nB - {neighbor_left(pixel, 0)} - {len(neighbor_left(pixel, 0))}")  # Пример вывода массива

    pass
    # header, img_data = read_bmp(filename)

    # header.print_fields()

if __name__ == "__main__":
    main()
