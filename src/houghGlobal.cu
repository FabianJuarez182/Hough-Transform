#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// GPU kernel. One thread per image pixel is spawned.
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;  // Limitar el acceso a hilos válidos

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <input_image.pgm>\n", argv[0]);
        return -1;
    }

    PGMImage inImg(argv[1]);
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float *pcCos = (float *) malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *) malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    unsigned char *d_in;
    int *d_hough;
    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    float *d_Cos, *d_Sin;
    cudaMalloc((void **) &d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **) &d_Sin, sizeof(float) * degreeBins);
    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    int blockNum = ceil((float)(w * h) / 256);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel Execution Time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int *h_hough = (int *) malloc(sizeof(int) * degreeBins * rBins);
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    PGMImage outImg(w, h, 255);
    int threshold = 100; // ajustable

    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            if (h_hough[rIdx * degreeBins + tIdx] > threshold) {
                for (int x = 0; x < w; x++) {
                    int y = (int)((rIdx * rScale - rMax - x * pcCos[tIdx]) / pcSin[tIdx]);
                    if (y >= 0 && y < h) {
                        outImg.pixels[y * w + x] = 255; // Marca las líneas en blanco
                    }
                }
            }
        }
    }

    outImg.write("output.pgm");

    cudaFree(d_in);
    cudaFree(d_hough);
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    free(h_hough);
    free(pcCos);
    free(pcSin);

    printf("Done! Output written to output.pgm\n");

    return 0;
}
