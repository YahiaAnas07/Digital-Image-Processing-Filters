# Image Processing Project

## Description
This project is an implementation of various image processing techniques using Python. It includes functions for noise filtering, transform/frequency domain filters, spatial domain filters, sharpening filters, and more.

## Features
- Noise filters: Uniform noise, Gaussian noise, Impulse noise
- Transform/Frequency domain filters: Fourier transform, Inverse Fourier transform, Fourier spectrum, Fourier shift
- Spatial domain filters: Median filter, Average filter, Adaptive median filter, Adaptive min filter, Gaussian filter
- Sharpening filters: Roberts sharpening, Sobel sharpening, Laplacian sharpening, Unsharp masking high-boost filtering
- Image compression using Huffman coding

## Installation
1. Clone the repository:
    ```
    git clone <repository_url>
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Import the necessary functions from the provided modules:
    ```python
    from image_processing import *
    ```
2. Apply the desired image processing techniques to your images.

## Examples
Here are some examples demonstrating how to use the functions provided in this project:
- Applying a median filter:
    ```python
    img_median_filtered = median_filter(image, filter_size=3)
    ```
- Compressing an image using Huffman coding:
    ```python
    compressed_image = compress_image(image)
    ```
- Applying Fourier transform and inverse Fourier transform:
    ```python
    img_fourier = Fourier(image)
    img_inverse_fourier = InverseFourier(img_fourier)
    ```

## License
This project is licensed under the [MIT License](LICENSE).

## Author
- Yahia Mohamed Anas
- Seifeldin Amr
- Ahmed Abdelmoneim
- Aly Essam
