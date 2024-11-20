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

// Memoria constante para senos y cosenos
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// GPU kernel
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            // Se utiliza memoria constante para calculos trigoonometricos
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

    // valores trigonométricos a memoria constante
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in;
    int *d_hough;
    int *h_hough = (int *) malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Medición de tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int blockNum = ceil((float)(w * h) / 256);
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel Execution Time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Validación simple con la CPU
    int *cpuht;
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);
    for (int i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i]) {
            printf("Calculation mismatch at: %i %i %i\n", i, cpuht[i], h_hough[i]);
        }
    }
    printf("Done!\n");

    // Liberar memoria
    cudaFree(d_in);
    cudaFree(d_hough);
    free(h_hough);
    free(pcCos);
    free(pcSin);

    return 0;
}
