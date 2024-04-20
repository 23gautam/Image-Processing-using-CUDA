#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <math.h>

#define MASK_WIDTH 16
#define TILE_WIDTH 16
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float mask[MASK_WIDTH * MASK_WIDTH];

__global__ void blurKernel(float *inputImage, float *outputImage, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_WIDTH + ty;

    // Load data into shared memory
    __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH];
    if (x < width && y < height) {
        tile[ty][tx] = inputImage[y * width + x];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    float result = 0.0f;
    int count = 0;
    if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
        for (int i = -MASK_WIDTH / 2; i <= MASK_WIDTH / 2; ++i) {
            for (int j = -MASK_WIDTH / 2; j <= MASK_WIDTH / 2; ++j) {
                int idx = ty + i;
                int idy = tx + j;
                if (idx >= 0 && idx < BLOCK_WIDTH && idy >= 0 && idy < BLOCK_WIDTH) {
                    result += tile[idx][idy];
                    count++;
                }
            }
        }
        result /= count;

        if (x < width && y < height) {
            outputImage[y * width + x] = result;
        }
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

void blurImage(float *inputImage, float *outputImage, int width, int height, float *mask_h) {
    float *inputImage_d, *outputImage_d, *mask_d;
    size_t size = width * height * sizeof(float);
    size_t maskSize = MASK_WIDTH * MASK_WIDTH * sizeof(float);

    cudaMalloc((void **)&inputImage_d, size);
    cudaMalloc((void **)&outputImage_d, size);
    cudaMalloc((void **)&mask_d, maskSize);

    cudaMemcpy(inputImage_d, inputImage, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, mask_h, maskSize);

    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blurKernel<<<dimGrid, dimBlock>>>(inputImage_d, outputImage_d, width, height);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Execution Time: %f milliseconds\n", milliseconds);

    cudaMemcpy(outputImage, outputImage_d, size, cudaMemcpyDeviceToHost);

    cudaFree(inputImage_d);
    cudaFree(outputImage_d);
    cudaFree(mask_d);
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

    // Initialize a simple blur filter
    float mask_h[MASK_WIDTH * MASK_WIDTH] = {1.0f / 9, 1.0f / 9, 1.0f / 9,
                                              1.0f / 9, 1.0f / 9, 1.0f / 9,
                                              1.0f / 9, 1.0f / 9, 1.0f / 9};

    // Allocate memory for the outputImage array
    float *outputImage = (float *)malloc(width * height * sizeof(float));

    blurImage(inputImage, outputImage, width, height, mask_h);

    // Now outputImage contains the blurred image
    //  saveImage(outputImage, width, height, "output_image.bmp");

    saveImage(outputImage, width, height, "output_image_blur.png");
    // Free allocated memory
    free(inputImage);
    free(outputImage);
    stbi_image_free(image_data);

    return 0;
}
