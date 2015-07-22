#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cv;

#define BLOCK_SIZE 10
#define RANGE 4

#define THRESHOLD 100

#define DEBUG_VECTORS 1
#define DEBUG_GRID 2
#define DEBUG_WEIGHTS 4
#define DEBUG_PARTICLES 8
#define DEBUG_NONZEROS 16
#define PARTICLE_WIDTH 5
#define PARTICLE_INFLUENCE 50
#define VECTOR_WEIGHT 0.3f

__device__ float roundFloat(float num, float low, float high)
{
  float comparator = (high + low) / 2.0f;
  if(num >= comparator)
    return high;
  return low;
}

__device__ void MotionEstimation3(uchar* src1, uchar* src2, int blockXBegin, int blockYBegin, int* devReturnX,
                    int* devReturnY, float* devReturnTotal, float* sharedArrayTotal, int* sharedArrayVR, int* sharedArrayVC, int sharedArraySize, int col, int row, int totalRows, int totalCols, int imageStep, int imageSizeX, int imageSizeY)
{
  int idx = threadIdx.x;

  int minNum = 0x7f800000;
  int minRow = 0;
  int minCol = 0;

  sharedArrayTotal[idx] = 0;
  sharedArrayVR[idx] = 0;
  sharedArrayVC[idx] = 0;

  int numCols = BLOCK_SIZE - RANGE;
  int numRows = BLOCK_SIZE - RANGE;

  int total = numCols * numRows;
  // if(blockIdx.x == 185 && threadIdx.x == 0)
  //   printf("%d %d\n", row, col);
  int ind = idx;

  int i,j;
  int centerX = BLOCK_SIZE * col + BLOCK_SIZE / 2;
  int centerY = BLOCK_SIZE * row + BLOCK_SIZE / 2;

  while(ind < total)
  {
    //printf("%d\n", ind);
    int startX = ind % numCols;
    int startY = (ind - startX) / numCols;
    float totalError1 = 0.0f;
    float totalError2 = 0.0f;

    float totalError = abs(totalError1 - totalError2);
    for(i = 0; i < RANGE; i++)
    {
      for(j = 0; j < RANGE; j++)
      {
        int src1Ind = imageStep * row * BLOCK_SIZE + col * BLOCK_SIZE + imageStep * (startY + j) + (startX + i);
        int src2Ind = (centerY - (RANGE / 2) + j) * imageStep + (centerX - (RANGE / 2) + i);

        if(i == 0 && j == 0 && blockIdx.x == 185 && threadIdx.x == 0 && ind == 0)
        {

        }

        float temp1, temp2;
        temp1 = (float)src1[src1Ind];
        temp2 = (float)src2[src2Ind];

        // temp1 = roundFloat(temp1, 0.0f, 0xFF);
        // temp2 = roundFloat(temp2, 0.0f, 0xFF);



        totalError1 += temp1;
        totalError2 += temp2;

        totalError += abs(temp1 - temp2);

      }
    }


    if(totalError < minNum)
    {

        // if(threadIdx.x == 0 && blockIdx.x == 185)
        //   printf("%d %f\n", blockIdx.x, totalError);
        minRow = startY - (BLOCK_SIZE - RANGE) / 2;// * 3 / 4;
        minCol = startX - (BLOCK_SIZE - RANGE) / 2;
        minNum = totalError;
    }

    ind += blockDim.x;

  }

  sharedArrayTotal[idx] = minNum;
  sharedArrayVR[idx] = minRow;// - 3 * RANGE / 4;
  sharedArrayVC[idx] = minCol;// - 3 * RANGE / 4;

  __syncthreads();


  if(threadIdx.x == 0)
  {
    int vr, vc;

    minNum = 0x7f800000;

    for(i = 1; i < sharedArraySize; i++)
    {
      if(sharedArrayTotal[i] < minNum)
      {

        vr = sharedArrayVR[i];
        vc = sharedArrayVC[i];


        minNum = sharedArrayTotal[i];
      }
    }

    devReturnX[col * totalRows + row] = vc;
    devReturnY[col * totalRows + row] = vr;
    devReturnTotal[col * totalRows + row] = minNum;
  }


}

__global__ void computeImageChange3(uchar* src1, uchar* src2, int* devReturnX, int* devReturnY, float* devReturnTotal, int imageStep, int imageSizeX, int imageSizeY, int totalRows, int totalCols, int sharedArraySize) {
  extern __shared__ float sharedArray[];

  float* sharedArrayTotal = (float*)&sharedArray[0];
  int* sharedArrayVR      = (int*)&sharedArray[sharedArraySize];
  int* sharedArrayVC     = (int*)&sharedArray[sharedArraySize * 2];

  //int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x;

  int i = blockIdx.x;
  int total = totalRows * totalCols;
  if(blockIdx.x == 0 && threadIdx.x == 0)
  {
    //printf("%d %d\n", totalRows, totalCols);
  }
  while(i < total)
  {
    int col = i % totalRows;
    int row = (i - col) / totalRows;
    int blockXBegin, blockYBegin;
    blockXBegin = (col) * BLOCK_SIZE;
    blockYBegin = (row) * BLOCK_SIZE;
    MotionEstimation3(src1, src2, blockXBegin, blockYBegin, devReturnX, devReturnY, devReturnTotal, sharedArrayTotal, sharedArrayVR,
      sharedArrayVC, sharedArraySize, row, col, totalRows, totalCols, imageStep, imageSizeX, imageSizeY);
    __syncthreads();
    i += stride;
  }

}

__device__ void MotionEstimation2(uchar* src1, uchar* src2, int blockXBegin, int blockYBegin, int* devReturnX,
                    int* devReturnY, float* devReturnTotal, float* sharedArrayTotal, int* sharedArrayVR, int* sharedArrayVC, int sharedArraySize, int col, int row, int totalRows, int totalCols, int imageStep, int imageSizeX, int imageSizeY)
{
  int idx = threadIdx.x;

  int minNum = 0x7f800000;
  int minRow = 0;
  int minCol = 0;

  sharedArrayTotal[idx] = 0;
  sharedArrayVR[idx] = 0;
  sharedArrayVC[idx] = 0;

  int numCols = BLOCK_SIZE - RANGE;
  int numRows = BLOCK_SIZE - RANGE;

  int total = numCols * numRows;
  int ind = idx;

  int i,j;
  int centerX = BLOCK_SIZE * col + BLOCK_SIZE / 2;
  int centerY = BLOCK_SIZE * row + BLOCK_SIZE / 2;

  while(ind < total)
  {
    //printf("%d\n", ind);
    int startX = ind % numCols;
    int startY = (ind - startX) / numCols;
    float totalError1 = 0.0f;
    float totalError2 = 0.0f;

    float totalError = abs(totalError1 - totalError2);
    for(i = 0; i < RANGE; i++)
    {
      for(j = 0; j < RANGE; j++)
      {
        int src1Ind = imageStep * row * BLOCK_SIZE + col * BLOCK_SIZE + imageStep * (startY + j) + (startX + i);
        int src2Ind = (centerY - (RANGE / 2) + j) * imageStep + (centerX - (RANGE / 2) + i);

        if(i == 0 && j == 0 && blockIdx.x == 100 && threadIdx.x == 0 && ind == 0)
        {
          // printf("imageStep, centerY, centerX, startY, startX\n");
          // printf("%d, %d, %d, %d, %d\n", imageStep, centerY, centerX, startY, startX);

        }

        float temp1, temp2;
        temp1 = (float)src1[src1Ind];
        temp2 = (float)src2[src2Ind];

        // temp1 = roundFloat(temp1, 0.0f, 0xFF);
        // temp2 = roundFloat(temp2, 0.0f, 0xFF);



        totalError1 += temp1;
        totalError2 += temp2;

        totalError += abs(temp1 - temp2);

      }
    }


    if(totalError < minNum)
    {

        // if(threadIdx.x == 0 && blockIdx.x == 185)
        //   printf("%d %f\n", blockIdx.x, totalError);
        minRow = startY - RANGE * 3 / 4;
        minCol = startX - RANGE * 3 / 4;
        minNum = totalError;
    }

    ind += blockDim.x;

  }

  sharedArrayTotal[idx] = minNum;
  sharedArrayVR[idx] = minRow;// - 3 * RANGE / 4;
  sharedArrayVC[idx] = minCol;// - 3 * RANGE / 4;

  __syncthreads();


  if(threadIdx.x == 0)
  {
    int vr, vc;

    minNum = 0x7f800000;

    for(i = 1; i < sharedArraySize; i++)
    {
      if(sharedArrayTotal[i] < minNum)
      {

        vr = sharedArrayVR[i];
        vc = sharedArrayVC[i];


        minNum = sharedArrayTotal[i];
      }
    }

    devReturnX[col * totalRows + row] = vc;
    devReturnY[col * totalRows + row] = vr;
    devReturnTotal[col * totalRows + row] = minNum;
  }


}

__global__ void computeImageChange2(uchar* src1, uchar* src2, int* devReturnX, int* devReturnY, float* devReturnTotal, int imageStep, int imageSizeX, int imageSizeY, int totalRows, int totalCols, int sharedArraySize) {
  extern __shared__ float sharedArray[];

  float* sharedArrayTotal = (float*)&sharedArray[0];
  int* sharedArrayVR      = (int*)&sharedArray[sharedArraySize];
  int* sharedArrayVC     = (int*)&sharedArray[sharedArraySize * 2];

  //int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x;

  int i = blockIdx.x;
  int total = totalRows * totalCols;
  if(blockIdx.x == 0 && threadIdx.x == 0)
  {
    //printf("%d %d\n", totalRows, totalCols);
  }
  while(i < total)
  {
    int col = i % totalRows;
    int row = (i - col) / totalRows;
    int blockXBegin, blockYBegin;
    blockXBegin = (col) * BLOCK_SIZE;
    blockYBegin = (row) * BLOCK_SIZE;
    MotionEstimation2(src1, src2, blockXBegin, blockYBegin, devReturnX, devReturnY, devReturnTotal, sharedArrayTotal, sharedArrayVR,
      sharedArrayVC, sharedArraySize, row, col, totalRows, totalCols, imageStep, imageSizeX, imageSizeY);

    i += stride;
  }

}

__global__ void updateParticleLocations(int* devReturnX, int* devReturnY, float* devReturnTotal, float* devParticleLocX, float* devParticleLocY,
  float* devParticleVelX, float* devParticleVelY, int numOfParticles, int totalRows, int totalCols, int imageWidth, int imageHeight)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int ind = idx;
  while(ind < numOfParticles)
  {
    int i, j;
    float newVelX = devParticleVelX[ind];
    float newVelY = devParticleVelY[ind];
    float particleRange = PARTICLE_INFLUENCE * PARTICLE_INFLUENCE;
    float particleLocX = devParticleLocX[ind];
    float particleLocY = devParticleLocY[ind];

    // for(i = 0; i < numOfParticles; i++)
    // {
    //   if(i != idx)
    //   {
    //     float otherParticleLocX = devParticleLocX[i];
    //     float otherParticleLocY = devParticleLocY[i];
    //     float dist = (otherParticleLocX - particleLocX) * (otherParticleLocX - particleLocX) + (otherParticleLocY - particleLocY) * (otherParticleLocY - particleLocY);
    //     if(dist < particleRange)
    //     {
    //       float otherParticleVelX = devParticleVelX[i];
    //       float otherParticleVelY = devParticleVelY[i];
    //
    //       newVelX += otherParticleVelX * (1 - (dist / particleRange));
    //       newVelY += otherParticleVelY * (1 - (dist / particleRange));
    //     }
    //   }
    // }

    for(i = 0; i < totalRows; i++)
    {
      for(j = 0; j < totalCols; j++)
      {
        float vectorLocX = j * BLOCK_SIZE + BLOCK_SIZE / 2;
        float vectorLocY = i * BLOCK_SIZE + BLOCK_SIZE / 2;
        float dist = (vectorLocX - particleLocX) * (vectorLocX - particleLocX) + (vectorLocY - particleLocY) * (vectorLocY - particleLocY);
        if(dist < particleRange && devReturnTotal[j * totalRows + i] > THRESHOLD)
        {
          float vectorVelX = devReturnX[j * totalRows + i];
          float vectorVelY = devReturnY[j * totalRows + i];

          newVelX += vectorVelX * (1 - (dist / particleRange)) * VECTOR_WEIGHT;
          newVelY += vectorVelY * (1 - (dist / particleRange)) * VECTOR_WEIGHT;

        }
      }
    }
    newVelX *= 0.9f;
    newVelY *= 0.9f;
    if(abs(newVelX) < 0.01f)
      newVelX = 0.0f;
    if(abs(newVelY) < 0.01f)
      newVelY = 0.0f;


    devParticleLocX[ind] += newVelX;
    devParticleLocY[ind] += newVelY;
    devParticleVelX[ind] = newVelX;
    devParticleVelY[ind] = newVelY;


    //run checks for image boundaries
    particleLocX = devParticleLocX[ind];
    particleLocY = devParticleLocY[ind];
    if(particleLocX < 0 || particleLocX > imageWidth)
    {
      if(particleLocX < 0)
        particleLocX = 0;
      else
        particleLocX = imageWidth;
      devParticleVelX[ind] *= -1;
    }

    if(particleLocY < 0 || particleLocY > imageHeight)
    {
      if(particleLocY < 0)
        particleLocY = 0;
      else
        particleLocY = imageHeight;
      devParticleVelY[ind] *= -1;
    }

    devParticleLocX[ind] = particleLocX;
    devParticleLocY[ind] = particleLocY;


    ind += stride;
  }

}

__device__ void MotionEstimation1(uchar* src1, uchar* src2, int blockXBegin, int blockYBegin, float* devReturnX,
                    float* devReturnY, int row, int col, int totalRows, int totalCols, int imageStep, int imageSizeX, int imageSizeY)
{
  int indexRowsUp =    max(0 + (row - 1) * BLOCK_SIZE - RANGE, 0);
  int indexRowsDown =  min(0 + (row - 1) * BLOCK_SIZE + RANGE, imageSizeY - BLOCK_SIZE);

  int indexColsLeft =  max(0 + (col - 1) * BLOCK_SIZE - RANGE, 0);
  int indexColsRight = min(0 + (col - 1) * BLOCK_SIZE + RANGE, imageSizeX - BLOCK_SIZE);

  int numRows = indexRowsDown - indexRowsUp + 1;
  int numCols = indexColsRight - indexColsLeft + 1;

  int nr, nc;

  float minNum = 0x7f800000;
  float minRow = 0;
  float minCol = 0;
  for(nr = 0; nr < numRows;nr++)
  {
    for(nc = 0; nc < numCols; nc++)
    {
      int srcXBegin, srcXEnd, srcYBegin, srcYEnd;

      srcYBegin = indexRowsUp + nr - 1;
      srcYEnd = indexRowsUp + nr - 1 + BLOCK_SIZE - 1;
      srcXBegin = indexColsLeft + nc - 1;
      srcXEnd = indexColsLeft + nc - 1 + BLOCK_SIZE - 1;

      int total = 0;
      int i,j;
      for(i = 0; i < srcYEnd - srcYBegin; i++)
      {
        for(j = 0; j < srcXEnd - srcXBegin; j++)
        {  int srcTempX = srcXBegin + j;
          int srcTempY = srcYBegin + i;
          uchar temp1, temp2;
          temp1 = src1[srcTempY * imageStep + srcTempX];
          temp2 = src2[(blockYBegin + i) * imageStep + (blockXBegin + j)];

          float tempf1, tempf2;
          tempf1 = (float)temp1 / (float)0xFF;
          tempf2 = (float)temp2 / (float)0xFF;

          tempf1 = min(max(tempf1, 0.0f), 1.0f);
          tempf2 = min(max(tempf2, 0.0f), 1.0f);

          if(tempf1 < 0.5)
          {
            tempf1 = 0;
          }
          else
          {
            tempf1 = 1;
          }


          if(tempf2 < 0.5)
          {
            tempf2 = 0;
          }
          else
          {
            tempf2 = 1;
          }

          total += abs(temp1 - temp2);
        }
      }

      if(total < minNum)
      {
          minNum = total;
          minRow = nr;
          minCol = nc;
      }
    }
  }

  int vr = minRow;
  int vc = minCol;
  vr = indexRowsUp +  vr - 1 - ((0 + (row - 1)) * BLOCK_SIZE);
  vc = indexColsLeft +  vc - 1 - ((0 + (col - 1)) * BLOCK_SIZE);

  devReturnX[row * numCols + col] = vc;
  devReturnY[row * numCols + col] = vr;
}

__global__ void computeImageChange1(uchar* src1, uchar* src2, float* devReturnX, float* devReturnY, int imageStep, int imageSizeX, int imageSizeY, int rows, int cols) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x;

  int i = idx;
  int total = rows * cols;
  while(i < total)
  {
    int row = i % cols;
    int col = (i - row) / cols;
    int blockXBegin, blockYBegin;
    blockXBegin = 1 + (col - 1) * BLOCK_SIZE;
    blockYBegin = 1 + (row - 1) * BLOCK_SIZE;
    MotionEstimation1(src1, src2, blockXBegin, blockYBegin, devReturnX, devReturnY, row, col, rows, cols, imageStep, imageSizeX, imageSizeY);

    i += stride;
  }

}
void detectAndDraw( gpu::GpuMat& dev_src1, gpu::GpuMat& dev_src2, Mat& src, int nr, int nc, float scale, int debugFlags,
                    float* devParticleLocX, float* devParticleLocY, float* devParticleVelX, float* devParticleVelY, int numOfParticles);
Mat getFrame(CvCapture*);

int main( int argc, const char** argv )
{
  CvCapture* capture = 0;
  Mat frame, lastFrame, image;
  String inputName;

  float scale = 1.0f;

  capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );

  cout << "before named window" << endl;
  cvNamedWindow( "result", 1 );
  cout << "after" << endl;

  cout << "In capture ..." << endl;

  gpu::GpuMat dev_lastFrame;

  frame = getFrame(capture);

  Mat grayframe;
  Size size = frame.size();
  int nr = size.height / BLOCK_SIZE;
  int nc = size.width / BLOCK_SIZE;
  cvtColor(frame, grayframe, CV_RGB2GRAY);
  frame.copyTo(lastFrame);
  dev_lastFrame.upload(grayframe);
  int debugFlags = 0;

  float* devParticleLocX;
  float* devParticleLocY;
  float* devParticleVelX;
  float* devParticleVelY;

  int numOfParticleCols = (size.width - PARTICLE_WIDTH) / PARTICLE_WIDTH;
  int numOfParticleRows = (size.height - PARTICLE_WIDTH) / PARTICLE_WIDTH;
  int numOfParticles = numOfParticleRows * numOfParticleCols;
  cudaMalloc((void**)&devParticleLocX, sizeof(float) * numOfParticles);
  cudaMalloc((void**)&devParticleLocY, sizeof(float) * numOfParticles);

  int i, j;
  float* hostParticleLocX;
  float* hostParticleLocY;

  hostParticleLocX = (float*)malloc(sizeof(float) * numOfParticles);
  hostParticleLocY = (float*)malloc(sizeof(float) * numOfParticles);

  for(i = 0; i < numOfParticleCols; i++)
  {
    for(j = 0; j < numOfParticleRows; j++)
    {
      float X = i * PARTICLE_WIDTH + PARTICLE_WIDTH / 2;
      float Y = j * PARTICLE_WIDTH + PARTICLE_WIDTH / 2;

      hostParticleLocX[i * numOfParticleRows + j] = X;
      hostParticleLocY[i * numOfParticleRows + j] = Y;
    }
  }

  cudaMemcpy(devParticleLocX, hostParticleLocX, sizeof(float) * numOfParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(devParticleLocY, hostParticleLocY, sizeof(float) * numOfParticles, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&devParticleVelX, sizeof(float) * numOfParticles);
  cudaMalloc((void**)&devParticleVelY, sizeof(float) * numOfParticles);
  cudaMemset(devParticleVelX, 0, sizeof(float) * numOfParticles);
  cudaMemset(devParticleVelY, 0, sizeof(float) * numOfParticles);
  cudaDeviceSynchronize();

  // free(hostParticleLocX);
  // free(hostParticleLocY);
  //debugFlags ^= DEBUG_PARTICLES;
  debugFlags ^= DEBUG_VECTORS;

  for(;;)
  {

    gpu::GpuMat dev_currFrame;

    frame = getFrame(capture);

    cvtColor(frame, grayframe, CV_RGB2GRAY);

    dev_currFrame.upload(grayframe);
    detectAndDraw(dev_currFrame, dev_lastFrame, lastFrame, nr, nc, scale, debugFlags, devParticleLocX, devParticleLocY, devParticleVelX, devParticleVelY, numOfParticles);
    frame.copyTo(lastFrame);

    dev_lastFrame = dev_currFrame;
    int key = waitKey(10);
    if(key == '1')
      debugFlags ^= DEBUG_GRID;
    else if(key == '2')
      debugFlags ^= DEBUG_VECTORS;
    else if (key == '3')
      debugFlags ^= DEBUG_WEIGHTS;
    else if (key == '4')
      debugFlags ^= DEBUG_PARTICLES;
    else if (key == '5')
    {
        cudaMemcpy(devParticleLocX, hostParticleLocX, sizeof(float) * numOfParticles, cudaMemcpyHostToDevice);
        cudaMemcpy(devParticleLocY, hostParticleLocY, sizeof(float) * numOfParticles, cudaMemcpyHostToDevice);
        cudaMemset(devParticleVelX, 0, sizeof(float) * numOfParticles);
        cudaMemset(devParticleVelY, 0, sizeof(float) * numOfParticles);
    }
    else if (key == '6')
      debugFlags ^= DEBUG_NONZEROS;
    else if( key >= 0 )
      goto _cleanup_;

  }

  //waitKey(0);

_cleanup_:
  cvReleaseCapture( &capture );

  cvDestroyWindow("result");

  return 0;
}

Mat getFrame(CvCapture* capture)
{
  Mat frameCopy;
  IplImage* iplImg = cvQueryFrame( capture );
  Mat frame = iplImg;

  if( iplImg->origin == IPL_ORIGIN_TL )
      frame.copyTo( frameCopy );
  else
      flip( frame, frameCopy, 90 );
  return frameCopy;
}


string getImageType(int number)
{
  // find type
  int imgTypeInt = number%8;
  std::string imgTypeString;

  switch (imgTypeInt)
  {
    case 0:
      imgTypeString = "8U";
      break;
    case 1:
      imgTypeString = "8S";
      break;
    case 2:
      imgTypeString = "16U";
      break;
    case 3:
      imgTypeString = "16S";
      break;
    case 4:
      imgTypeString = "32S";
      break;
    case 5:
      imgTypeString = "32F";
      break;
    case 6:
      imgTypeString = "64F";
      break;
    default:
      break;
  }

  // find channel
  int channel = (number/8) + 1;

  stringstream type;
  type<<"CV_"<<imgTypeString<<"C"<<channel;

  return type.str();
}

static void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color,
int thickness=1, int line_type=8, int shift=0, double tipLength=0.1)
{
    const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
    line(img, pt1, pt2, color, thickness, line_type, shift);
    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
    Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
    cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
    line(img, p, pt2, color, thickness, line_type, shift);
    p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
}


void detectAndDraw( gpu::GpuMat& dev_src1, gpu::GpuMat& dev_src2, Mat& src, int nr, int nc, float scale, int debugFlags,
                    float* devParticleLocX, float* devParticleLocY, float* devParticleVelX, float* devParticleVelY, int numOfParticles)
{

  int* devReturnX;
  int* devReturnY;
  float* devReturnTotal;

  cudaMalloc((void **)&devReturnX, sizeof(int) * nr * nc);
  cudaMalloc((void **)&devReturnY, sizeof(int) * nr * nc);
  cudaMalloc((void **)&devReturnTotal, sizeof(float) * nr * nc);
  cudaMemset(devReturnTotal, 0, sizeof(float) * nr * nc);
  cudaMemset(devReturnX, 0, sizeof(int) * nr * nc);
  cudaMemset(devReturnY, 0, sizeof(int) * nr * nc);
  cudaDeviceSynchronize();
  Size size = src.size();
  int width = size.width;
  int height = size.height;
  int numBlocks = (width * height) / (BLOCK_SIZE * BLOCK_SIZE);
  int threadsPerBlock = 256;// (BLOCK_SIZE - RANGE)*(BLOCK_SIZE - RANGE);
  dim3 block(numBlocks);
  dim3 thread(threadsPerBlock);

  computeImageChange3<<<block, thread,  threadsPerBlock * 3 * sizeof(float)>>>(dev_src1.data, dev_src2.data, devReturnX, devReturnY, devReturnTotal, dev_src1.step,
    width, height, nr, nc, threadsPerBlock);
  cudaDeviceSynchronize();
  if((debugFlags & DEBUG_PARTICLES) > 0){
    updateParticleLocations<<<block, thread>>>(devReturnX, devReturnY, devReturnTotal, devParticleLocX, devParticleLocY, devParticleVelX, devParticleVelY, numOfParticles, nr, nc, width, height);
    cudaDeviceSynchronize();
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    cout << cudaGetErrorString(err) << endl;
    waitKey(100);
  }
  int* hostReturnX = (int*)malloc(sizeof(int) * nr * nc);
  int* hostReturnY = (int*)malloc(sizeof(int) * nr * nc);
  float* hostReturnTotals = (float*)malloc(sizeof(int) * nr * nc);
  float* hostParticleLocX = (float*)malloc(sizeof(int) * numOfParticles);
  float* hostParticleLocY = (float*)malloc(sizeof(int) * numOfParticles);
  cudaMemcpy(hostReturnX, devReturnX, sizeof(int) * nr * nc, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostReturnY, devReturnY, sizeof(int) * nr * nc, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostReturnTotals, devReturnTotal, sizeof(float) * nr * nc, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostParticleLocX, devParticleLocX, sizeof(float) * numOfParticles, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostParticleLocY, devParticleLocY, sizeof(float) * numOfParticles, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int i,j;
  Mat output = src;

  if((debugFlags & DEBUG_PARTICLES) > 0)
  {
    for(i = 0; i < numOfParticles; i++)
    {
      float X = hostParticleLocX[i];
      float Y = hostParticleLocY[i];

      Point p1;
      p1.x = (int)X;
      p1.y = (int)Y;

      circle(output, p1, 2, Scalar(255,0,255), 1, 8, 0);
    }
  }

  for(i = 0; i < nr && debugFlags > 0; i++)
  {
    if((debugFlags & (DEBUG_GRID)) > 0)
    {
      Point p1,p2;
      p1.x = 0;
      p1.y = i * BLOCK_SIZE;
      p2.x = width;
      p2.y = i * BLOCK_SIZE;


      line(output, p1, p2, Scalar(0,0,255), 1, 8, 0);
    }

    for(j = 0; j < nc; j++)
    {

      Point start, end;

      start.x = j * BLOCK_SIZE + BLOCK_SIZE / 2;
      start.y = i * BLOCK_SIZE + BLOCK_SIZE / 2;

      int X, Y;


      float num = hostReturnTotals[j * nr + i];
      float tempNum = THRESHOLD;
      if(num < tempNum)
      {
        end.x = start.x;
        end.y = start.y;
      }
      else
      {
        X = hostReturnX[j * nr + i];
        Y = hostReturnY[j * nr + i];
        end.x = start.x + X * scale;
        end.y = start.y + Y * scale;
      }



      if(i == 0 && (debugFlags & (DEBUG_GRID)) > 0)
      {
        Point p1, p2;
        p1.x = j * BLOCK_SIZE;
        p1.y = 0;
        p2.x = j * BLOCK_SIZE;
        p2.y = height;

        line(output, p1, p2, Scalar(0,0,255), 1, 8, 0);
      }

      if((debugFlags & (DEBUG_GRID)) > 0)
      {
        circle(output, end, RANGE, Scalar(255,0,0), CV_FILLED, 8, 0);
      }

      if((debugFlags & (DEBUG_VECTORS)) > 0){
        if((debugFlags & (DEBUG_NONZEROS)) > 0){
          if(start.x != end.x || start.y != end.y)
            arrowedLine(output, start, end, Scalar(0,255,0));
        }
        else
          arrowedLine(output, start, end, Scalar(0,255,0));
      }
      if((debugFlags & (DEBUG_WEIGHTS)) > 0)
      {
        ostringstream ss;
        ss << num;
        string s(ss.str());
        putText(output, s, start, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,255,255), 1, 16);

      }
    }



  }
  Point t1;
  t1.x = BLOCK_SIZE / 2;
  t1.y = 2 * BLOCK_SIZE;


  cudaFree(dev_src2.data);
  cudaFree(devReturnX);
  cudaFree(devReturnY);
  free(hostReturnX);
  free(hostReturnY);
  imshow( "result", output);
  //getchar();
}
