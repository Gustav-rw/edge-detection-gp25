
#include "gpu_kernel.cu"
#include "support.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  Timer timer;

  // Initialize host variables ----------------------------------------------

  printf("\nSetting up the problem...");
  fflush(stdout);
  startTime(&timer);

  unsigned char *I_h, *O_h;
  unsigned char *I_d, *O_d;
  unsigned int imageHeight, imageWidth;
  cudaError_t cuda_ret;

  /* Read image dimensions */
  if (argc == 1) {
    imageWidth = 1920;
    imageHeight = 1080;
  } else if (argc == 2) {
    imageWidth = atoi(argv[1]);
    imageHeight = atoi(argv[1]);
  } else if (argc == 3) {
    imageWidth = atoi(argv[1]);
    imageHeight = atoi(argv[2]);
  } else {
    printf("\n    Invalid input parameters!"
           "\n    Usage: ./convolution          # Image is 1920 x 1080"
           "\n    Usage: ./convolution <m>      # Image is m x m"
           "\n    Usage: ./convolution <m> <n>  # Image is m x n"
           "\n");
    exit(0);
  }

  unsigned int img_sz = imageHeight * imageWidth;

  /* Allocate host memory */
  I_h = (unsigned char*) malloc( sizeof(char)*img_sz );
  O_h = (unsigned char*) malloc( sizeof(char)*img_sz );

  //generate random input image
  srand(time(NULL));
  for (unsigned int i=0; i < img_sz; i++) { I_h[i] = (char) (rand()%256); }


  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));
  printf("    Image: %u x %u\n", imageWidth, imageHeight);

  // Allocate device variables ----------------------------------------------

  printf("Allocating device variables...");
  fflush(stdout);
  startTime(&timer);

  cudaMalloc((void **) &I_d, sizeof(char) * img_sz);
  cudaMalloc((void **) &O_d, sizeof(char) * img_sz);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Copy host variables to device ------------------------------------------

  printf("Copying data from host to device...");
  fflush(stdout);
  startTime(&timer);

  /* Copy image to device global memory */
  cudaMemcpy(I_d, I_h, sizeof(char)*img_sz, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Launch kernel ----------------------------------------------------------
  printf("Launching kernel...");
  fflush(stdout);
  startTime(&timer);

  // INSERT CODE HERE

  dim3 numThreadsPerBlock(BLOCK_DIM+2, BLOCK_DIM+2);
  dim3 numBlocks((imageWidth + BLOCK_DIM - 1)/BLOCK_DIM, \
                 (imageHeight + BLOCK_DIM - 1)/BLOCK_DIM);

  //in second argument imageHeight is divided by the number of images being processed to get the individual image frameHeight (1080)
  sobelGPU_Optimized <<< numBlocks, numThreadsPerBlock  >>> (imageWidth, (unsigned int)imageHeight/4, imageHeight, I_d, O_d, 200);

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess) {
    printf("Unable to launch/execute kernel");
    return 1;
  }

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Copy device variables from host ----------------------------------------

  printf("Copying data from device to host...");
  fflush(stdout);
  startTime(&timer);

  cudaMemcpy(O_h, O_d, sizeof(char)*img_sz, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Verify correctness -----------------------------------------------------

  printf("Verifying results...");
  fflush(stdout);

  //just checking that non-edges exist randomly within the output frame
  int i = 0;
  char yy = 0;
  while (i < img_sz) {
    if (O_h[i] == 0 && i % 1920 && i>1920 && (i+1) % 1920 && i < 500000) {
      printf("index %d", i);
      fflush(stdout);
      yy++;
      if (yy > 5) { break; }
    }
    i++;
  }

  // Free memory ------------------------------------------------------------

  free(I_h);
  free(O_h);

  cudaFree(I_d);
  cudaFree(O_d);

  return 0;
}
