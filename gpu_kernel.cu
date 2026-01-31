#define BLOCK_DIM 14 //dimension of the block of pixels actually being sobel-calculated

/*
* Descr: GPU sobel convolution; edges are set to 0
*   Common pixels for the block is loaded into shared memory    
* Args:
*   frameWidth, frameHeight: give the dimensions of the individual input and output frames
*   totalHeight: a multiple of frameHeight, for batch processing
*   grayFrame: input frame
*   edgeMask: output frame
*   edgeThreshold: threshold for pixel being considered an edge after sobel filter
* Comments:
*   The function should be called by two-dimensional blocks of dimension [BLOCK_DIM + 2]
*/
__global__ void sobelGPU_Optimized(unsigned int frameWidth, unsigned int frameHeight, unsigned int totalHeight, unsigned char* grayFrame, unsigned char* edgeMask, short edgeThreshold) {

    //shared memory tile for input pixels
    __shared__ unsigned char sharedTile[BLOCK_DIM + 2][BLOCK_DIM + 2];

    //get thread row and column
    //unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    //unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    //calculate the row and column indexes of the pixel loaded into shared memory by this thread
    signed int pixelRow = blockIdx.y * BLOCK_DIM + threadIdx.y;
    signed int pixelCol = blockIdx.x * BLOCK_DIM + threadIdx.x;

    //thread loads its input pixel into shared memory, checking boundary conditions
    if (pixelRow < totalHeight && pixelCol < frameWidth) {
        sharedTile[threadIdx.y][threadIdx.x] = grayFrame[pixelRow*frameWidth + pixelCol];
    }

    __syncthreads();

    //perform sobel convolution for the thread, checking boundary conditions
    if (threadIdx.y > 0 && threadIdx.x > 0 && //thread should not be in extension area
        threadIdx.y < BLOCK_DIM + 1 && threadIdx.x < BLOCK_DIM + 1 && //thread should not be in extension area
        pixelRow < totalHeight && pixelCol < frameWidth) { //thread's pixel must exist in total image

        //if thread's pixel is on the edge of an individual image: the sobel result is automatically 0
        if (pixelRow % frameHeight == 0 || pixelCol < 1 || (pixelRow+1) % frameHeight == 0 || pixelCol >= frameWidth-1) {
            edgeMask[pixelRow*frameWidth + pixelCol] = 0;
        }
        else
        {
            //calculate gx and gy parts separately

            short gx = -   sharedTile[threadIdx.y-1][threadIdx.x-1] +
                           sharedTile[threadIdx.y-1][threadIdx.x+1] +
                       -2* sharedTile[threadIdx.y][threadIdx.x-1]   +
                        2* sharedTile[threadIdx.y][threadIdx.x+1]   +
                       -   sharedTile[threadIdx.y+1][threadIdx.x-1] +
                           sharedTile[threadIdx.y+1][threadIdx.x+1];

            short gy = -   sharedTile[threadIdx.y-1][threadIdx.x-1] +
                       -2* sharedTile[threadIdx.y-1][threadIdx.x]   +
                       -   sharedTile[threadIdx.y-1][threadIdx.x+1] +
                           sharedTile[threadIdx.y+1][threadIdx.x-1] +
                        2* sharedTile[threadIdx.y+1][threadIdx.x]   +
                           sharedTile[threadIdx.y+1][threadIdx.x+1];

            //calculate final result of sobel
            short res = (short) __fsqrt_rn(gx*gx + gy*gy); //rounded square root

            //place result in output image
            edgeMask[pixelRow*frameWidth + pixelCol] = (res >= edgeThreshold) ? 255 : 0;
        }
    }

}
