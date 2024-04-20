#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Include time.h for timing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MASK_WIDTH 7

void blurImage(float *inputImage, float *outputImage, int width, int height, float *mask) {
    int maskRadius = MASK_WIDTH / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            for (int my = -maskRadius; my <= maskRadius; ++my) {
                for (int mx = -maskRadius; mx <= maskRadius; ++mx) {
                    int sx = x + mx;
                    int sy = y + my;
                    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                        sum += inputImage[sy * width + sx] * mask[(my + maskRadius) * MASK_WIDTH + (mx + maskRadius)];
                    }
                }
            }
            outputImage[y * width + x] = sum;
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

    // Timer variables
    clock_t start, end;
    double cpu_time_used;

    // Start the timer
    start = clock();

    blurImage(inputImage, outputImage, width, height, mask_h);

    // Stop the timer
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %f milliseconds\n", cpu_time_used*1000);

    // Now outputImage contains the blurred image
    saveImage(outputImage, width, height, "output_image_blur_serial.png");

    // Free allocated memory
    free(inputImage);
    free(outputImage);
    stbi_image_free(image_data);

    return 0;
}
