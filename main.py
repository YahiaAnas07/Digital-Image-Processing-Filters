import heapq
import math
import pickle
from collections import Counter

import cv2
import gradio as gr
import numpy as np
from PIL import Image


class Node:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


def build_huffman_tree(frequency):
    heap = [Node(symbol, freq) for symbol, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.frequency + node2.frequency)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


def generate_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        if node.left:
            generate_huffman_codes(node.left, prefix + "0", codebook)
        if node.right:
            generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook


def compress_image(image):
    # Calculate the frequency of each pixel value
    pixel_values = image.flatten()
    frequency = Counter(pixel_values)

    huffman_tree = build_huffman_tree(frequency)
    huffman_codes = generate_huffman_codes(huffman_tree)
    flat_image = image.flatten()
    encoded_image = ''.join([huffman_codes[pixel] for pixel in flat_image])
    compressed_image = compress_image(image)

    decompress_image(encoded_image, compressed_image, huffman_codes, image.shape)
    return encoded_image


def decompress_image(encoded_image, compressed_image, huffman_codes, shape):
    with open('compressed_image.bin', 'wb') as file:
        byte_array = int(compressed_image, 2).to_bytes((len(compressed_image) + 7) // 8, byteorder='big')
        file.write(byte_array)

    with open('huffman_codes.pkl', 'wb') as file:
        pickle.dump(huffman_codes, file)

    with open('compressed_image.bin', 'rb') as file:
        byte_array = file.read()
        compressed_image = bin(int.from_bytes(byte_array, byteorder='big'))[2:]

    with open('huffman_codes.pkl', 'rb') as file:
        huffman_codes = pickle.load(file)

    inverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_image = []
    current_code = ""

    # Process each bit in the encoded image
    for bit in encoded_image:
        current_code += bit
        if current_code in inverse_huffman_codes:
            decoded_image.append(inverse_huffman_codes[current_code])
            current_code = ""

    # Calculate expected length from the shape
    expected_length = np.prod(shape)
    decoded_length = len(decoded_image)
    print(f"Length of decoded image: {decoded_length}")
    print(f"Expected number of elements: {expected_length}")

    # Adjust the length of decoded_image to match expected length
    if decoded_length < expected_length:
        # Pad with zeros if decoded_image is too short
        decoded_image.extend([0] * (expected_length - decoded_length))
    elif decoded_length > expected_length:
        # Truncate if decoded_image is too long
        decoded_image = decoded_image[:expected_length]

    # Reshape the decoded image to the original shape
    return np.array(decoded_image).reshape(shape)


######################################
########### Noise Filters ############
######################################
def uniformNoise(image, a, b):
    noise = np.array(image, copy=True)
    a, b = (int)(a), (int)(b)
    for x in range(noise.shape[0]):
        for y in range(noise.shape[1]):
            for c in range(noise.shape[2]):
                if a <= image[x, y, c] <= b:
                    rnd = np.random.random()
                    noise_value = a + rnd * (b - a)
                    noise[x, y, c] = noise_value

    return Image.fromarray(noise)


def gaussianNoise(image, mean=0, std=0.01):
    mean = float(mean)
    std = float(std)
    image = np.array(image, copy=True)
    h, w, c = image.shape

    noise = np.array(image, dtype=float)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                u1 = np.random.random()
                u2 = np.random.random()
                z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                noise_value = mean + z0 * std
                noise[i, j, k] = noise_value

    # Add the noise to the image
    noisy_image = image + (noise * 100)

    # Clip the values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert NumPy array back to PIL Image
    noisy_image = Image.fromarray(noisy_image.astype(np.uint8))

    return noisy_image


def impulseNoise(image, prob=0.1):
    noise = np.array(image, copy=True)
    width, height, c = noise.shape
    prob = (float)(prob)
    for x in np.arange(width):
        for y in np.arange(height):
            rnd = (float)(np.random.random())
            if rnd < prob:
                rnd = (float)(np.random.random())
                if rnd > 0.5:
                    noise[x, y] = 255
                else:
                    noise[x, y] = 0

    return Image.fromarray(noise)


############################################
#### Transform/Frequency Domain Filters ####
############################################
def doHistogram(img):
    histogram = [0 for _ in range(256)]
    flat = img.flatten()
    for pixel in flat:
        histogram[pixel] += 1

    histogram
    accHistogram = [0 for _ in range(256)]
    accHistogram[0] = histogram[0]
    for i in range(1, 256):
        accHistogram[i] = accHistogram[i - 1] + histogram[i]
    return accHistogram


def equHistogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    accHistogram = doHistogram(img)
    width = img.shape[0]
    height = img.shape[1]
    v = img.shape[-1]
    eqImg = np.zeros_like(img)
    n = width * height
    for i in range(256):
        accHistogram[i] = np.floor(255 * accHistogram[i] / n)
    for r in range(width):
        for c in range(height):
            q = img[r, c]
            eqImg[r, c] = np.floor((accHistogram[q]))

    return eqImg


def histMatching(mainImg, pdfImg):
    h1 = doHistogram(mainImg)
    h2 = doHistogram(pdfImg)
    width = mainImg.shape[0]
    height = mainImg.shape[1]
    n = width / height
    eqImg = np.zeros_like(mainImg)
    for i in range(256):
        h1[i] = np.floor(255 * h1[i] / n)
    for i in range(256):
        h2[i] = np.floor(255 * h2[i] / n)
    map = np.zeros(256, dtype=int)
    for i in range(256):
        map[i] = np.argmin(np.abs(h1[i] - h2))
    for r in range(width):
        for c in range(height):
            q = mainImg[r, c]
            eqImg[r, c] = map[q]
    return eqImg


def nearestNeighborInterpolation(img, nw, nh):
    nw = int(nw)
    nh = int(nh)
    width = img.shape[0]
    height = img.shape[1]
    co = img.shape[-1]
    newWidth = width / nw
    newHeight = height / nh
    nImg = np.zeros((nw, nh, 3), dtype=img.dtype)
    for i in range(co):
        for r in range(nw):
            for c in range(nh):
                rr = int(np.floor(r * newWidth))
                cc = int(np.floor(c * newHeight))
                nImg[r, c, i] = img[rr, cc, i]
    return nImg


def bilinearInterpolation(img, nw, nh):
    nw = int(nw)
    nh = int(nh)
    width = img.shape[0]
    height = img.shape[1]
    co = img.shape[-1]
    newWidth = width / nw
    newHeight = height / nh
    nImg = np.zeros((nw, nh, 3), dtype=img.dtype)
    for i in range(co):
        for r in range(nw):
            for c in range(nh):
                rr = int(np.floor(r * newWidth))
                cc = int(np.floor(c * newHeight))
                dx = r * newWidth - rr
                dy = c * newHeight - cc
                rr = min(rr, width - 2)
                cc = min(cc, height - 2)
                v1 = img[rr, cc, i]
                v2 = img[rr + 1, cc, i]
                v3 = img[rr, cc + 1, i]
                v4 = img[rr + 1, cc + 1, i]
                a = v1 * (1 - dx) * (1 - dy) + v2 * dx * (1 - dy) + v3 * (1 - dx) * dy + v4 * dx * dy
                nImg[r, c, i] = a
    return nImg


def CalcW(x):
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a + 2) * (pos_x ** 3)) - ((a + 3) * (pos_x ** 2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x ** 3)) - (5 * a * (pos_x ** 2)) + (8 * a * pos_x) - 4 * a)
    else:
        return 0


def bicubicInterpolation(img, nw, nh):
    nw = int(nw)
    nh = int(nh)

    output = np.zeros((nw, nh, img.shape[2]), dtype=np.uint8)
    for c in range(img.shape[2]):
        for i in range(nw):
            for j in range(nh):
                xm = (i + 0.5) * (img.shape[0] / nw) - 0.5
                ym = (j + 0.5) * (img.shape[1] / nh) - 0.5
                xi = np.floor(xm)
                yi = np.floor(ym)
                u = xm - xi
                v = ym - yi
                x = [int(xi) - 1, int(xi), int(xi) + 1, int(xi) + 2]
                y = [int(yi) - 1, int(yi), int(yi) + 1, int(yi) + 2]
                if ((x[0] >= 0) and (x[3] < img.shape[0]) and (y[0] >= 0) and (y[3] < img.shape[1])):
                    dist_x0 = CalcW(x[0] - xm)
                    dist_x1 = CalcW(x[1] - xm)
                    dist_x2 = CalcW(x[2] - xm)
                    dist_x3 = CalcW(x[3] - xm)
                    dist_y0 = CalcW(y[0] - ym)
                    dist_y1 = CalcW(y[1] - ym)
                    dist_y2 = CalcW(y[2] - ym)
                    dist_y3 = CalcW(y[3] - ym)
                    out = (img[x[0], y[0], c] * (dist_x0 * dist_y0) +
                           img[x[0], y[1], c] * (dist_x0 * dist_y1) +
                           img[x[0], y[2], c] * (dist_x0 * dist_y2) +
                           img[x[0], y[3], c] * (dist_x0 * dist_y3) +
                           img[x[1], y[0], c] * (dist_x1 * dist_y0) +
                           img[x[1], y[1], c] * (dist_x1 * dist_y1) +
                           img[x[1], y[2], c] * (dist_x1 * dist_y2) +
                           img[x[1], y[3], c] * (dist_x1 * dist_y3) +
                           img[x[2], y[0], c] * (dist_x2 * dist_y0) +
                           img[x[2], y[1], c] * (dist_x2 * dist_y1) +
                           img[x[2], y[2], c] * (dist_x2 * dist_y2) +
                           img[x[2], y[3], c] * (dist_x2 * dist_y3) +
                           img[x[3], y[0], c] * (dist_x3 * dist_y0) +
                           img[x[3], y[1], c] * (dist_x3 * dist_y1) +
                           img[x[3], y[2], c] * (dist_x3 * dist_y2) +
                           img[x[3], y[3], c] * (dist_x3 * dist_y3))
                    output[i, j, c] = np.clip(out, 0, 255)
    return output


def FourierTransform2D(dImg):
    dImg = cv2.cvtColor(dImg, cv2.COLOR_BGR2GRAY)

    width, height = dImg.shape
    dfImg = np.zeros((width, height), dtype=np.complex128)

    for u in range(width):
        for v in range(height):
            f = 0
            for x in range(width):
                for y in range(height):
                    f += dImg[x, y] * np.exp(-2j * np.pi * ((u * x) / width + (v * y) / height))
            dfImg[u, v] = f

    return dfImg


def InverseFourierTransform2D(dImg):
    dImg = cv2.cvtColor(dImg, cv2.COLOR_BGR2GRAY)
    width, height = dImg.shape
    invImg = np.zeros((width, height), dtype=np.complex128)

    for x in range(width):
        for y in range(height):
            iF = 0
            for u in range(width):
                for v in range(height):
                    iF += dImg[u, v] * np.exp(2j * np.pi * ((u * x) / width + (v * y) / height))
            invImg[x, y] = iF / (width * height)

    return int(invImg)


def FourierSpectrum(spectrum):
    newSpectrum = np.log(1 + np.abs(spectrum))
    return newSpectrum


def FourierShift(arr):
    dims = arr.shape
    ci = []
    for dim in dims:
        ci.append(dim // 2)
    sa = np.roll(arr, ci, axis=(0, 1))
    return sa


def Fourier(img):
    fI = FourierTransform2D(img)
    sI = FourierShift(fI)
    magnitude_spectrum = FourierSpectrum(sI)
    magnitude_spectrum = 255 * (magnitude_spectrum - magnitude_spectrum.min()) / (
            magnitude_spectrum.max() - magnitude_spectrum.min())
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    return magnitude_spectrum


def InverseFourier(img):
    fI = FourierTransform2D(img)
    sI = FourierShift(img)
    spec = FourierSpectrum(sI)
    magnitude_spectrum = FourierSpectrum(sI)
    magnitude_spectrum = 255 * (magnitude_spectrum - magnitude_spectrum.min()) / (
            magnitude_spectrum.max() - magnitude_spectrum.min())
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    return magnitude_spectrum


##############################################
############ Sharpening Filters ##############
##############################################
def robertsSharpening(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img.shape
    # already flipped
    robertsX = np.array([[-1, 0],
                         [0, 1, ]])
    robertsY = np.array([[0, -1],
                         [1, 0]])
    pSize = 1
    imgx = np.zeros((m, n))
    imgy = np.zeros((m, n))
    pImg = np.zeros((m + 2 * pSize, n + 2 * pSize))
    pImg[pSize:-pSize, pSize:-pSize] = img
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = (pImg[i - 1, j - 1] * robertsX[0, 0] +
                    pImg[i - 1, j] * robertsX[0, 1] +
                    pImg[i, j - 1] * robertsX[1, 0] +
                    pImg[i, j] * robertsX[1, 1])

            imgx[i - pSize, j - pSize] = temp

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = (pImg[i - 1, j - 1] * robertsY[0, 0] +
                    pImg[i - 1, j] * robertsY[0, 1] +
                    pImg[i, j - 1] * robertsY[1, 0] +
                    pImg[i, j] * robertsY[1, 1])

            imgy[i - pSize, j - pSize] = temp

    magnitude = imgx + imgy
    magnitude = np.array(magnitude, dtype=np.uint8)
    return magnitude + np.array(img, dtype=np.uint8)


def sobelSharpening(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img.shape
    # both filter are already flipped and ready for conv
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    pSize = 1
    imgx = np.zeros((m, n))
    imgy = np.zeros((m, n))
    pImg = np.zeros((m + 2 * pSize, n + 2 * pSize))
    pImg[pSize:-pSize, pSize:-pSize] = img
    for i in range(1, m):
        for j in range(1, n):
            temp = pImg[i - 1, j - 1] * sobel_y[0, 0] + pImg[i - 1, j] * sobel_y[0, 1] + pImg[
                i - 1, j + 1] * sobel_y[0, 2] + img[i, j - 1] * sobel_y[1, 0] + pImg[i, j] * sobel_y[
                       1, 1] + pImg[i, j + 1] * sobel_y[1, 2] + pImg[i + 1, j - 1] * sobel_y[2, 0] + pImg[
                       i + 1, j] * sobel_y[2, 1] + pImg[i + 1, j + 1] * sobel_y[2, 2]
            imgy[i - pSize, j - pSize] = temp

    for i in range(1, m):
        for j in range(1, n):
            temp = pImg[i - 1, j - 1] * sobel_x[0, 0] + pImg[i - 1, j] * sobel_x[0, 1] + pImg[
                i - 1, j + 1] * sobel_x[0, 2] + pImg[i, j - 1] * sobel_x[1, 0] + pImg[i, j] * sobel_x[
                       1, 1] + pImg[i, j + 1] * sobel_x[1, 2] + pImg[i + 1, j - 1] * sobel_x[2, 0] + pImg[
                       i + 1, j] * sobel_x[2, 1] + pImg[i + 1, j + 1] * sobel_x[2, 2]
            imgx[i - pSize, j - pSize] = temp

    magnitude = imgx + imgy
    return np.uint8(magnitude) + np.uint8(img)


def laplacianSharpening(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img.shape
    # flipping the filter wont chnage its output
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    pSize = 1
    imgN = np.zeros((m, n))
    pImg = np.zeros((m + 2 * pSize, n + 2 * pSize))
    pImg[pSize:-pSize, pSize:-pSize] = img

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = pImg[i - 1, j - 1] * laplacian_kernel[0, 0] + pImg[i - 1, j] * laplacian_kernel[0, 1] + \
                   pImg[i - 1, j + 1] * laplacian_kernel[0, 2] + pImg[i, j - 1] * laplacian_kernel[1, 0] + \
                   pImg[i, j] * laplacian_kernel[1, 1] + pImg[i, j + 1] * laplacian_kernel[1, 2] + pImg[
                       i + 1, j - 1] * laplacian_kernel[2, 0] + pImg[i + 1, j] * laplacian_kernel[2, 1] + \
                   pImg[i + 1, j + 1] * laplacian_kernel[2, 2]

            imgN[i - pSize, j - pSize] = temp

    return np.array(imgN, dtype=np.uint8) + np.array(img, dtype=np.uint8)


def smoothingFilter(img):
    if img.ndim == 3 and img.shape[2] == 3:
        data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n, c = img.shape
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9
    pSize = 1
    imgN = np.zeros((m, n), dtype=float)
    pImg = np.zeros((m + 2 * pSize, n + 2 * pSize), dtype=np.uint8)
    pImg[pSize:-pSize, pSize:-pSize, :] = img

    for i in range(1, m):
        for j in range(1, n):
            temp = (pImg[i - 1, j - 1] * mask[0, 0] + pImg[i - 1, j] * mask[0, 1] + pImg[i - 1, j + 1] *
                    mask[0, 2] + pImg[i, j - 1] * mask[1, 0] + pImg[i, j] * mask[1, 1] + pImg[
                        i, j + 1] *
                    mask[1, 2] + pImg[i + 1, j - 1] * mask[2, 0] + pImg[i + 1, j] * mask[2, 1] + pImg[i + 1, j + 1] *
                    mask[2, 2])

            imgN[i - pSize, j - pSize] = temp

    imgN = np.array(imgN, dtype=int)

    return imgN


def unsharpMasking_highBoostFiltering(img, k):
    k = (int)(k)
    # if k>1 its High boost filtering
    blured = smoothingFilter(img)
    mask = img - blured
    img1 = img + (k * mask)
    img1 = np.clip(img1, 0, 255)
    return img1


#################################################
##############Spatial Domain Filters################
def median_filter(data, filter_size, padding_type='zero'):
    filter_size = (int)(filter_size)
    data = np.array(data)
    temp = []
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    indexer = filter_size // 2
    data_final = np.zeros_like(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            temp = []  # Clear the temporary list for each pixel
            for z in range(filter_size):
                for k in range(filter_size):
                    # Get the value of the pixel in the neighborhood
                    x_index = max(0, min(len(data) - 1, i + z - indexer))
                    y_index = max(0, min(len(data[0]) - 1, j + k - indexer))
                    temp.append(data[x_index, y_index])

            # Convert temp to a one-dimensional array before sorting
            temp = np.array(temp).flatten()
            temp.sort()
            data_final[i, j] = temp[len(temp) // 2]

    return data_final


def average_filter(data, filter_size, padding_type='zero'):
    filter_size = (int)(filter_size)
    temp = []
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    indexer = filter_size // 2  # Therefore if the filter size was 3 indexer will be 1
    data_final = np.zeros((len(data), len(
        data[0])))  # making another 2d matrix that initialized all with zeros so that we can put all the averages
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            total = 0
            count = 0
            for z in range(filter_size):
                for k in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        if padding_type == 'zero':
                            count += 1
                        elif padding_type == 'replicate':
                            total += data[max(0, min(len(data) - 1, i + z - indexer)), j]
                            count += 1
                    elif j + k - indexer < 0 or j + k - indexer > len(data[0]) - 1:
                        if padding_type == 'zero':
                            count += 1
                        elif padding_type == 'replicate':
                            total += data[i, max(0, min(len(data[0]) - 1, j + k - indexer))]
                            count += 1
                    else:
                        total += data[i + z - indexer, j + k - indexer]
                        count += 1
            data_final[i, j] = (int)(total) / (int)(count)

    return np.array(data_final, dtype=np.uint8)


def adaptive_median_filter(data, max_filter_size, padding_type='zero'):
    max_filter_size = (int)(max_filter_size)

    # Convert RGB to grayscale if needed
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    data_final = np.zeros_like(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filter_size = 3
            while filter_size <= max_filter_size:
                indexer = filter_size // 2
                temp = []

                for z in range(filter_size):
                    for k in range(filter_size):
                        ni = i + z - indexer
                        nj = j + k - indexer

                        if ni < 0 or ni >= len(data) or nj < 0 or nj >= len(data[0]):
                            if padding_type == 'zero':
                                temp.append(0)
                            elif padding_type == 'replicate':
                                ni = max(0, min(len(data) - 1, ni))
                                nj = max(0, min(len(data[0]) - 1, nj))
                                temp.append(data[ni, nj])
                        else:
                            temp.append(data[ni, nj])

                median_value = np.median(temp)

                # Debugging: print the median and filter values
                if i % 10 == 0 and j % 10 == 0:  # Print every 10th pixel to reduce output
                    print(f"Pixel ({i},{j}), Filter size: {filter_size}, Median: {median_value}, Temp: {temp}")

                if median_value != min(temp) and median_value != max(temp):
                    data_final[i, j] = median_value
                    break

                filter_size += 2

            if filter_size > max_filter_size:
                data_final[i, j] = median_value

    return data_final


def adaptive_min_filter(data, max_filter_size, padding_type='zero'):
    max_filter_size = (int)(max_filter_size)
    # Convert RGB to grayscale if needed
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data_final = np.zeros_like(data)
    rows, cols = data.shape

    for i in range(rows):
        for j in range(cols):
            filter_size = 3
            while filter_size <= max_filter_size:
                indexer = filter_size // 2
                min_val = float('inf')
                for z in range(filter_size):
                    for k in range(filter_size):
                        ni = i + z - indexer
                        nj = j + k - indexer

                        if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                            if padding_type == 'replicate':
                                ni = max(0, min(rows - 1, ni))
                                nj = max(0, min(cols - 1, nj))
                                val = data[ni, nj]
                            else:
                                continue  # For zero padding, ignore out of bounds
                        else:
                            val = data[ni, nj]

                        if val < min_val:
                            min_val = val

                # Break if min value found is different from the original pixel value
                if data[i, j] != min_val:
                    data_final[i, j] = min_val
                    break

                filter_size += 2

            # If the loop finished without finding a suitable min value, use the last min value
            if filter_size > max_filter_size:
                data_final[i, j] = min_val

    return data_final


def manual_sum(arr):
    total = 0
    for row in arr:
        for val in row:
            total += val
    return total


def apply_gaussian_filter(image, kernel):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    kernel = np.array(kernel)

    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2

    padded_image = np.pad(image, pad_width, mode='constant')

    output_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output_image[i, j] = manual_sum(region * kernel)
    return output_image


def gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size, size))
    center = size // 2
    sum_kernel = 0.0

    for x in range(size):
        for y in range(size):
            value = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
            kernel[x, y] = value
            sum_kernel += value

    kernel /= sum_kernel

    return kernel


def GaussianFilter(image, kernel, sigma):
    kernel = (int)(kernel)
    sigma = (int)(sigma)
    kernel = gaussian_kernel(kernel, sigma)
    smoothed_image = apply_gaussian_filter(image, kernel)
    return smoothed_image


#################################################
def scanString(Image, selectedOption, noiseOption, smoothingOption, sharpeningOption, transformOption, compression,
               spatialOption, param1, param2, param3):
    if selectedOption == "Noise Filters":
        if noiseOption == "Impulse noise":
            return impulseNoise(Image, param1)
        elif noiseOption == "Uniform noise":
            return uniformNoise(Image, param1, param2)
        elif noiseOption == "Gaussian noise":
            return gaussianNoise(Image, param1, param2)
    elif selectedOption == "Smoothing Filters":
        if smoothingOption == "Smoothing Filter":
            return smoothingFilter(Image)
    elif selectedOption == "Sharpening Filters":
        if sharpeningOption == "Roberts Sharpening":
            return robertsSharpening(Image)
        elif sharpeningOption == "Sobel Sharpening":
            return sobelSharpening(Image)
        elif sharpeningOption == "Laplacian Sharpening":
            return laplacianSharpening(Image)
        elif sharpeningOption == "UnsharpMasking HighBoost Filtering":
            return unsharpMasking_highBoostFiltering(Image, param1)
    elif selectedOption == "Spatial Filters":
        if spatialOption == "Gaussian Filter":
            return GaussianFilter(Image, param1, param2)
        elif spatialOption == "Median Filter":
            return median_filter(Image, param1)
        elif spatialOption == "Average Filter":
            return average_filter(Image, param1)
        elif spatialOption == "Adaptive Median Filter":
            return adaptive_median_filter(Image, param1)
        elif spatialOption == "Adaptive Min Filter":
            return adaptive_min_filter(Image, param1)
    elif selectedOption == "Transform Filters":
        if transformOption == "Equalized Histogram":
            return equHistogram(Image)
        elif transformOption == "Histogram Matching":
            return histMatching(Image, param3)
        elif transformOption == "Nearest Neighbor Interpolation":
            return nearestNeighborInterpolation(Image, param1, param2)
        elif transformOption == "Bilinear Interpolation":
            return bilinearInterpolation(Image, param1, param2)
        elif transformOption == "Bicubic Interpolation":
            return bicubicInterpolation(Image, param1, param2)
        elif transformOption == "Fourier Transform":
            return Fourier(Image)
        elif transformOption == "Inverse Fourier Transform":
            return InverseFourier(Image)
    elif selectedOption == "Compression Filters":
        if compression == "Hoffman Compression":
            return compress_image(Image)
    else:
        return Image


def update_visibility(noise_type):
    if noise_type == "Impulse noise":
        return gr.update(visible=True, label="Probability"), gr.update(visible=False), gr.update(visible=False)
    elif noise_type == "Gaussian noise":
        return gr.update(visible=True, label="Gaussian Mean"), gr.update(visible=True,
                                                                         label="Gaussian Std Dev"), gr.update(
            visible=False)
    elif noise_type == "Uniform noise":
        return gr.update(visible=True, label="Range start"), gr.update(visible=True, label="Range end"), gr.update(
            visible=False)
    elif noise_type == "UnsharpMasking HighBoost Filtering":
        return gr.update(visible=True, label='K'), gr.update(visible=False), gr.update(visible=False)
    elif noise_type == "Histogram Matching":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, label='PDF Image')
    elif noise_type == "Bilinear Interpolation":
        return gr.update(visible=True, label="New width"), gr.update(visible=True, label="New height"), gr.update(
            visible=False)
    elif noise_type == "Bicubic Interpolation":
        return gr.update(visible=True, label="New width"), gr.update(visible=True, label="New height"), gr.update(
            visible=False)
    elif noise_type == "Nearest Neighbor Interpolation":
        return gr.update(visible=True, label="New width"), gr.update(visible=True, label="New height"), gr.update(
            visible=False)
    elif noise_type == "Gaussian Filter":
        return gr.update(visible=True, label="Kernel"), gr.update(visible=True, label="Sigma"), gr.update(
            visible=False)
    elif noise_type == "Median Filter":
        return gr.update(visible=True, label="Filter size"), gr.update(visible=False), gr.update(
            visible=False)
    elif noise_type == "Average Filter":
        return gr.update(visible=True, label="Filter size"), gr.update(visible=False), gr.update(
            visible=False)
    elif noise_type == "Adaptive Median Filter":
        return gr.update(visible=True, label="Max Filter size"), gr.update(visible=False), gr.update(
            visible=False)
    elif noise_type == "Adaptive Min Filter":
        return gr.update(visible=True, label="Max Filter size"), gr.update(visible=False), gr.update(
            visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def update_visibility2(selected_filter):
    if selected_filter == "Smoothing Filters":
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))
    elif selected_filter == "Sharpening Filters":
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))
    elif selected_filter == "Noise Filters":
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))
    elif selected_filter == "Transform Filters":
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False))
    elif selected_filter == "Compression Options":
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False))
    elif selected_filter == "Spatial Filters":
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=True))
    else:
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False))


def clear_inputs():
    return None, None, None, None, None, None, None, None, None, None, None, None


theme = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="zinc",
    neutral_hue=gr.themes.Color(c100="#d1fae5", c200="#ffffff", c300="#6ee7b7", c400="#ffffff", c50="#ecfdf5",
                                c500="#10b981", c600="#059669", c700="#047857", c800="#065f46", c900="#064e3b",
                                c950="#054436"),
)

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        baseImage = gr.Image(label="Base Image")
        filterSelection = gr.Radio(
            choices=["Smoothing Filters", "Sharpening Filters", "Noise Filters", "Transform Filters",
                     "Compression Options", "Spatial Filters"], label="Select Filter Type")
        output = gr.Image(label="Result Image")
    with gr.Row():
        smoothingFilters = gr.Dropdown(choices=["Smoothing Filter"],
                                       label="Smoothing Filters", allow_custom_value=True, visible=False)
        sharpeningFilters = gr.Dropdown(choices=["Roberts Sharpening", "Sobel Sharpening", "Laplacian Sharpening",
                                                 "UnsharpMasking HighBoost Filtering"],
                                        label="Sharpening Filters", allow_custom_value=True, visible=False)
        noiseFilters = gr.Dropdown(choices=['Impulse noise', "Gaussian noise", "Uniform noise"], label="Noise Filters",
                                   allow_custom_value=True, visible=False)
        transformFilters = gr.Dropdown(
            choices=["Equalized Histogram", "Histogram Matching", "Nearest Neighbor Interpolation",
                     "Bilinear Interpolation", "Fourier Transform", "Inverse Fourier Transform",
                     "Bicubic Interpolation"],
            label="Transform/Frequency Domain Filters", allow_custom_value=True, visible=False)
        spatialFilters = gr.Dropdown(
            choices=["Gaussian Filter", "Average Filter", "Median Filter", "Adaptive Median Filter",
                     "Adaptive Min Filter"],
            label="Spatial Filters", allow_custom_value=True, visible=False)
        compressionOptions = gr.Dropdown(choices=['Hoffman Compression'],
                                         label="Compression Techniques", allow_custom_value=True, visible=False)
    with gr.Row():
        param1 = gr.Text(label='Parameter 1', visible=False)
        param2 = gr.Text(label='Parameter 2', visible=False)
        param3 = gr.Image(label='Parameter 3', visible=False)
    filterSelection.change(update_visibility2, inputs=filterSelection,
                           outputs=[smoothingFilters, sharpeningFilters, noiseFilters, transformFilters,
                                    compressionOptions, spatialFilters])

    noiseFilters.change(update_visibility, inputs=noiseFilters, outputs=[param1, param2, param3])
    smoothingFilters.change(update_visibility, inputs=smoothingFilters, outputs=[param1, param2, param3])
    sharpeningFilters.change(update_visibility, inputs=sharpeningFilters, outputs=[param1, param2, param3])
    transformFilters.change(update_visibility, inputs=transformFilters, outputs=[param1, param2, param3])
    compressionOptions.change(update_visibility, inputs=compressionOptions, outputs=[param1, param2, param3])
    spatialFilters.change(update_visibility, inputs=spatialFilters, outputs=[param1, param2, param3])

    with gr.Row():
        button = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear")

    button.click(scanString, inputs=[baseImage, filterSelection, noiseFilters, smoothingFilters, sharpeningFilters,
                                     transformFilters, compressionOptions, spatialFilters, param1, param2, param3],
                 outputs=output)
    clear.click(clear_inputs, outputs=[baseImage, filterSelection, param1, param2, param3, output, smoothingFilters,
                                       sharpeningFilters, noiseFilters, spatialFilters, transformFilters,
                                       compressionOptions])

    # Launch the Gradio demo
    demo.launch()
