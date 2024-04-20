#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // For timing
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MASK_WIDTH 3

float sobel_x[MASK_WIDTH][MASK_WIDTH] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

float sobel_y[MASK_WIDTH][MASK_WIDTH] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

void sobelEdgeDetection(float *inputImage, float *outputImage, int width, int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
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

    // Start the timer
    clock_t start = clock();

    // Call the Sobel edge detection function
    sobelEdgeDetection(inputImage, outputImage, width, height);

    // Stop the timer
    clock_t end = clock();

    // Calculate the elapsed time in milliseconds
    double elapsed_time = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Execution Time: %.2f ms\n", elapsed_time);

    // Now outputImage contains the gradient magnitude
    // Save the result (e.g., using stb_image_write or other image libraries)
    saveImage(outputImage, width, height, "output_image_edge_serial.png");

    // Free allocated memory
    free(inputImage);
    free(outputImage);
    stbi_image_free(image_data);

    return 0;
}
