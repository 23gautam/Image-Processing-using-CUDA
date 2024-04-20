#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <math.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 16
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float sobel_x[MASK_WIDTH][MASK_WIDTH] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ float sobel_y[MASK_WIDTH][MASK_WIDTH] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

__global__ void sobelEdgeDetection(float *inputImage, float *outputImage, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float gradient_x = 0.0f;
        float gradient_y = 0.0f;

        for (int i = 0; i < MASK_WIDTH; ++i) {
            for (int j = 0; j < MASK_WIDTH; ++j) {
                int image_row = row - MASK_WIDTH / 2 + i;
                int image_col = col - MASK_WIDTH / 2 + j;

                if (image_row >= 0 && image_row < height && image_col >= 0 && image_col < width) {
                    gradient_x += inputImage[image_row * width + image_col] * sobel_x[i][j];
                    gradient_y += inputImage[image_row * width + image_col] * sobel_y[i][j];
                }
            }
        }

        outputImage[row * width + col] = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);
    }
}

void saveImage(float *outputImage, int width, int height, const char *filename) {
    // Scale pixel values back to [0, 255]
    unsigned char *image_data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    for (int i = 0; i < width * height; ++i) {
        image_data[i] = (unsigned char)(outputImage[i] * 255.0f);
    }

    // Write image using stb_image_write
    stbi_write_png(filename, width, height, 1, image_data, width);

    // Free allocated memory
    free(image_data);
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    // Load the image using stb_image
    int width, height, channels;
    unsigned char *image_data = stbi_load("input.jpeg", &width, &height, &channels, STBI_rgb);

    if (!image_data) {
        printf("Error: Couldn't load image.\n");
        return -1;
    }

    // Allocate memory for inputImage array
    float *inputImage = (float *)malloc(width * height * sizeof(float));

    // Convert image data to grayscale and normalize pixel values
    for (int i = 0; i < width * height; ++i) {
        float pixel_value = (float)image_data[i * channels] / 255.0f; // Assuming RGB image, take the red channel for grayscale
        inputImage[i] = pixel_value;
    }

    // Allocate memory for the outputImage array
    float *outputImage = (float *)malloc(width * height * sizeof(float));
    float *d_inputImage, *d_outputImage;

    // Allocate device memory
    cudaMalloc(&d_inputImage, width * height * sizeof(float));
    cudaMalloc(&d_outputImage, width * height * sizeof(float));

    // Copy input image to device memory
    cudaMemcpy(d_inputImage, inputImage, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Call the Sobel edge detection kernel
    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelEdgeDetection<<<dimGrid, dimBlock>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %.2f ms\n", milliseconds);

    // Copy result back to host memory
    cudaMemcpy(outputImage, d_outputImage, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Now outputImage contains the gradient magnitude
    // Save the result (e.g., using stb_image_write or other image libraries)
    saveImage(outputImage, width, height, "output_image_edge.png");

    // Free allocated memory
    free(inputImage);
    free(outputImage);
    stbi_image_free(image_data);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}