#include <stdio.h>
#include<cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>

//given the float pixel index (x,y), return the bilinearInterpolation in the image
//directly write in funtion is faster than use this function
__device__ inline float3  bilinearInterpolation(unsigned char* imageDevice, float x, float y, int W, int H, int C){
	const int intx = (int)x, inty = (int)y;
	const float dx = x - intx, dy = y - inty;
	const int ptr = (intx + inty * W) * C;

	// 4 pixel
	const unsigned char* p00 = &imageDevice[ptr];
	const unsigned char* p01 = p00 + C;
	const unsigned char* p10 = p00 + W * C;
	const unsigned char* p11 = p00 + (W + 1) * C;
	
	// bilinearInterpolation
	float interpolated_b, interpolated_g, interpolated_r;
	interpolated_b = (1 - dx) * (1 - dy) * p00[0] + dx * (1 - dy) * p01[0] + 
		(1 - dx) * dy * p10[0] + dx * dy * p11[0];
	interpolated_g = (1 - dx) * (1 - dy) * p00[1] + dx * (1 - dy) * p01[1] + 
		(1 - dx) * dy * p10[1] + dx * dy * p11[1];
	interpolated_r = (1 - dx) * (1 - dy) * p00[2] + dx * (1 - dy) * p01[2] + 
		(1 - dx) * dy * p10[2] + dx * dy * p11[2];
	
	return make_float3(interpolated_b, interpolated_g, interpolated_r);
}

__global__ void undistortCudaKernel(unsigned char* inputDevice, unsigned char* outputDevice, int W, int H, int C)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= W || v >= H)
		return;

	// cameraMtx, cameraDis.  refer to python code
	const float fx = 3.08757924e+03, cx = 1.85563844e+03, fy = 3.08812875e+03, cy = 1.04865193e+03;
	const float k1 = 2.58505201e-02, k2 = 2.42460868e-01, p1 = 1.71551700e-03,
		p2 = -2.13479176e-05, k3 = -5.55736234e-01;

	float x = (u - cx) / fx, y = (v - cy) / fy;
	float r = sqrt(x * x + y * y);
	float x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r + k3 * r * r * r * r * r * r)
		+ 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
	float y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r + k3 * r * r * r * r * r * r)
		+ p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
	float u_distorted = fx * x_distorted + cx;
	float v_distorted = fy * y_distorted + cy;

	int undis_ptr = (u + v * W) * C;

	if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < W - 1 && v_distorted < H - 1) {
		float3 interpolated_bgr = bilinearInterpolation(inputDevice, u_distorted, v_distorted, W, H, C);
		outputDevice[undis_ptr] = (unsigned char)interpolated_bgr.x;
		outputDevice[undis_ptr + 1] = (unsigned char)interpolated_bgr.y;
		outputDevice[undis_ptr + 2] = (unsigned char)interpolated_bgr.z;
	}
	else {
		outputDevice[undis_ptr] = 0;
		outputDevice[undis_ptr + 1] = 0;
		outputDevice[undis_ptr + 2] = 0;
	}
}

__global__ void homographyCudaKernel(unsigned char* inputDevice, unsigned char* outputDevice, int W, int H, int C, int index) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= W || v >= H)
		return;
	//invH,  refer to python code
	float h0, h1, h2, h3, h4, h5, h6, h7, h8, h9;
	if (index == 1) {
		h0 = -9.72582017e-01, h1 = -5.85615183e-02, h2 = 3.76670660e+03;
		h3 = 3.97139534e-02, h4 = -1.00936537e+00, h5 = 2.08839174e+03;
		h6 = 1.13087711e-05, h7 = -1.56472913e-05, h8 = 9.92388040e-01;	
	}
	else if (index == 2) {
		h0 = -9.59934711e-01, h1 = -6.37100026e-02, h2 = 3.79281139e+03;
		h3 = 5.79134991e-02, h4 = -9.94511041e-01, h5 = 2.06483454e+03;
		h6 = 1.29483224e-05, h7 = -9.78119689e-06, h8 = 9.73285735e-01;
	}
	else if (index == 3) {
		h0 = 1.01777437e+00, h1 = 3.96991996e-03, h2 = -3.07761027e+01;
		h3 = -9.94219055e-03, h4 = 9.96584576e-01, h5 = 4.25557085e+01;
		h6 = 7.14568703e-06, h7 = -7.52802696e-06, h8 = 9.99463567e-01;	
	}
	else if (index == 4) {
		h0 = 1.03166196e+00, h1 = -1.21727271e-03, h2 = -3.29221982e+01;
		h3 = -3.20584301e-03, h4 = 1.00151135e+00, h5 = -1.26685190e+01;
		h6 = 1.11417640e-05, h7 = -9.19935294e-06, h8 = 9.99761585e-01;
	}

	float x = h0 * u + h1 * v + h2;
	float y = h3 * u + h4 * v + h5;
	float z = h6 * u + h7 * v + h8;
	x /= z;
	y /= z;
	int outPtr = (u + v * W) * C;

	if (x >= 0 && y >= 0 && x < W - 1 && y < H - 1) {
		float3 interpolated_bgr = bilinearInterpolation(inputDevice, x, y, W, H, C);
		outputDevice[outPtr] = interpolated_bgr.x;
		outputDevice[outPtr + 1] = interpolated_bgr.y;
		outputDevice[outPtr + 2] = interpolated_bgr.z;
	}
	else {
		outputDevice[outPtr] = 0;
		outputDevice[outPtr + 1] = 0;
		outputDevice[outPtr + 2] = 0;
	}
}

// input to output
__global__ void nothing(unsigned char* outputDevice, unsigned char* inputDevice, int W, int H, int C) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= W || v >= H)
		return;
	int ptr = (u + v * W) * C;
	outputDevice[ptr] = (unsigned char)inputDevice[ptr];
	outputDevice[ptr+1] = (unsigned char)inputDevice[ptr+1];
	outputDevice[ptr+2] = (unsigned char)inputDevice[ptr+2];
}

// (W,H)*4 images to (cutW *2,cutH*2) encoded image
__global__ void encodeImageCudaKernel(unsigned char* encodeDevice, unsigned char* inputDevice1, 
	unsigned char* inputDevice2, unsigned char* inputDevice3, unsigned char* inputDevice4, 
	int W, int H, int C, int cutW, int cutH, int shift) {

	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= cutW*2 || v >= cutH*2)
		return;
	const int gtcut1 = W / 2 - cutW / 2, gtcut2 = W / 2 + cutW / 2;
	const int gtcut3 = H / 2 - cutH / 2, gtcut4 = H / 2 + cutH / 2;
	const int encodePtr = (u + v * cutW * 2) * C; //encodeW=cghW*2

	const int viewIndexX = u % 2, viewIndexY = v % 2; //view 
	const int pixelIndexX = u / 2, pixelIndexY = v / 2; //pixel
	if (viewIndexX == 0 && viewIndexY == 0) { //view1
		const int imagePtr = (pixelIndexX + gtcut1 + shift / 2 + (pixelIndexY + gtcut3 + shift / 2) * W) * C;
		encodeDevice[encodePtr] = inputDevice1[imagePtr];
		encodeDevice[encodePtr + 1] = inputDevice1[imagePtr + 1];
		encodeDevice[encodePtr + 2] = inputDevice1[imagePtr + 2];
	}
	else if (viewIndexX == 1 && viewIndexY == 0) {//view2
		const int imagePtr = (pixelIndexX + gtcut1 - shift / 2 + (pixelIndexY + gtcut3 + shift / 2) * W) * C;
		encodeDevice[encodePtr] = inputDevice2[imagePtr];
		encodeDevice[encodePtr + 1] = inputDevice2[imagePtr + 1];
		encodeDevice[encodePtr + 2] = inputDevice2[imagePtr + 2];
	}
	else if (viewIndexX == 0 && viewIndexY == 1) {//view3
		const int imagePtr = (pixelIndexX + gtcut1 + shift / 2 + (pixelIndexY + gtcut3 - shift / 2) * W) * C;
		encodeDevice[encodePtr] = inputDevice3[imagePtr];
		encodeDevice[encodePtr + 1] = inputDevice3[imagePtr + 1];
		encodeDevice[encodePtr + 2] = inputDevice3[imagePtr + 2];
	}
	else if (viewIndexX == 1 && viewIndexY == 1) {//view4
		const int imagePtr = (pixelIndexX + gtcut1 - shift / 2 + (pixelIndexY + gtcut3 - shift / 2) * W) * C;
		encodeDevice[encodePtr] = inputDevice4[imagePtr];
		encodeDevice[encodePtr + 1] = inputDevice4[imagePtr + 1];
		encodeDevice[encodePtr + 2] = inputDevice4[imagePtr + 2];
	}
}

__global__ void calMapCudaKernel(float2* map, int W, int H, int C, int cutW, int cutH, int shift) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= cutW * 2 || v >= cutH * 2)
		return;
	const int gtcut1 = W / 2 - cutW / 2, gtcut2 = W / 2 + cutW / 2;
	const int gtcut3 = H / 2 - cutH / 2, gtcut4 = H / 2 + cutH / 2;
	const int mapPtr = (u + v * cutW * 2); //encodeW=cghW*2

	// cameraMtx, cameraDis.  refer to python code
	const float fx = 3.08757924e+03, cx = 1.85563844e+03, fy = 3.08812875e+03, cy = 1.04865193e+03;
	const float k1 = 2.58505201e-02, k2 = 2.42460868e-01, p1 = 1.71551700e-03,
		p2 = -2.13479176e-05, k3 = -5.55736234e-01;
	//invH,  refer to python code
	float h0, h1, h2, h3, h4, h5, h6, h7, h8, h9;
	int imageX = 0, imageY = 0;
	map[mapPtr].x = 0.001f;
	map[mapPtr].y = 0.001f;  // zero 

	const int viewIndexX = u % 2, viewIndexY = v % 2; //view 
	const int pixelIndexX = u / 2, pixelIndexY = v / 2; //pixel
	if (viewIndexX == 0 && viewIndexY == 0) { //view1
		imageX = pixelIndexX + gtcut1 + shift / 2;
		imageY = pixelIndexY +gtcut3 + shift / 2;
		h0 = -9.22245423e-01, h1 = -4.88943088e-02, h2 = 3.61666452e+03;
		h3 = 4.61380040e-02, h4 = -9.86303931e-01, h5 = 1.98371101e+03;
		h6 = 2.54671274e-05, h7 = -1.39722319e-05, h8 = 9.33673441e-01;

	}
	else if (viewIndexX == 1 && viewIndexY == 0) {//view2
		imageX = pixelIndexX + gtcut1 - shift / 2;
		imageY = pixelIndexY + gtcut3 + shift / 2;
		h0 = -8.96458195e-01, h1 = -5.26713060e-02, h2 = 3.60390886e+03;
		h3 = 6.53796237e-02, h4 = -9.62946282e-01, h5 = 1.94409813e+03;
		h6 = 2.91206446e-05, h7 = -8.24226463e-06, h8 = 9.05651011e-01;

	}
	else if (viewIndexX == 0 && viewIndexY == 1) {//view3
		imageX = pixelIndexX + gtcut1 + shift / 2;
		imageY = pixelIndexY + gtcut3 - shift / 2;
		h0 = 1.08903102e+00; h1 = 1.53709625e-02; h2 = -1.49780080e+02;
		h3 = -2.53095691e-03; h4 = 1.03342046e+00; h5 = 5.16166113e+01;
		h6 = 2.62595465e-05; h7 = -9.14213928e-06; h8 = 9.95916471e-01;
	}
	else if (viewIndexX == 1 && viewIndexY == 1) {//view4
		imageX = pixelIndexX + gtcut1 - shift / 2;
		imageY = pixelIndexY + gtcut3 - shift / 2;
		h0 = 1.09657379e+00; h1 = 3.64544003e-03; h2 = -1.14999789e+02;
		h3 = 1.13054485e-02; h4 = 1.03708472e+00; h5 = -9.21005537e+00;
		h6 = 2.85023588e-05; h7 = -9.39605722e-06; h8 = 9.97084341e-01;
	}
	float x = h0 * imageX + h1 * imageY + h2;
	float y = h3 * imageX + h4 * imageY + h5;
	float z = h6 * imageX + h7 * imageY + h8;
	x /= z;
	y /= z;
	if (x >= 0 && y >= 0 && x < W - 1 && y < H - 1) {
		x = (x - cx) / fx;
		y = (y - cy) / fy;
		const float r = sqrt(x * x + y * y);
		const float x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r + k3 * r * r * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
		const float y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r + k3 * r * r * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
		const float u_distorted = fx * x_distorted + cx;
		const float v_distorted = fy * y_distorted + cy;
		if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < W - 1 && v_distorted < H - 1) {
			map[mapPtr].x = u_distorted;
			map[mapPtr].y = v_distorted;

		}
	}
}

__global__ void applyMapCudaKernel(unsigned char* encodeDevice, unsigned char* inputDevice1,
	unsigned char* inputDevice2, unsigned char* inputDevice3, unsigned char* inputDevice4, 
	float2* map, int W, int H, int C,int cutW, int cutH) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= cutW * 2 || v >= cutH*2)
		return;

	const int viewIndexX = u % 2, viewIndexY = v % 2; //view 

	const int encodePtr = (u + v * cutW * 2) * C; //encodeW=cghW*2
	const int mapPtr = (u + v * cutW * 2);
	float3 interpolated_bgr;
	if (map[mapPtr].x == 0.001f) {
		encodeDevice[encodePtr] = (unsigned char)0;
		encodeDevice[encodePtr + 1] = (unsigned char)0;
		encodeDevice[encodePtr + 2] = (unsigned char)0;
	}
	else {
		if (viewIndexX == 0 && viewIndexY == 0) { //view1
			interpolated_bgr = bilinearInterpolation(inputDevice1, map[mapPtr].x, map[mapPtr].y, W, H, C);
		}
		else if (viewIndexX == 1 && viewIndexY == 0) {//view2
			interpolated_bgr = bilinearInterpolation(inputDevice2, map[mapPtr].x, map[mapPtr].y, W, H, C);
		}
		else if (viewIndexX == 0 && viewIndexY == 1) {//view3
			interpolated_bgr = bilinearInterpolation(inputDevice3, map[mapPtr].x, map[mapPtr].y, W, H, C);
		}
		else if (viewIndexX == 1 && viewIndexY == 1) {//view4
			interpolated_bgr = bilinearInterpolation(inputDevice4, map[mapPtr].x, map[mapPtr].y, W, H, C);
		}
		encodeDevice[encodePtr] = (unsigned char)interpolated_bgr.x;
		encodeDevice[encodePtr + 1] = (unsigned char)interpolated_bgr.y;
		encodeDevice[encodePtr + 2] = (unsigned char)interpolated_bgr.z;
	}
}

// srgbToLinear
__device__ inline float srgbToLinear(float value) {
	if (value <= 0.04045f)
		return value / 12.92f;
	else
		return powf((value + 0.055f) / 1.055f, 2.4f);
}

// convert uchar image to float network input
// image
//BGR BGR BGR BGR BGR BGR
//network
//BBB BBB GGG GGG RRR RRR
//and srgb to linear, sqrt
__global__ void imageToNetworkKernel(unsigned char* imageInput, float* networkInput, int W, int H, int C) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= W  || v >= H )
		return;

	int imageIndex = (u + v * W ) * C;

	networkInput[(0 * H  + v) * W  + u] = sqrtf(srgbToLinear(imageInput[imageIndex] / 255.0f)); // B
	networkInput[(1 * H  + v) * W  + u] = sqrtf(srgbToLinear(imageInput[imageIndex + 1] / 255.0f)); // G
	networkInput[(2 * H  + v) * W  + u] = sqrtf(srgbToLinear(imageInput[imageIndex + 2] / 255.0f)); // R

}

// phase to image(0-1)
// phase to image(0-1)
__device__ inline float phaseToImage(float phase, float offset) {
	float result = phase + offset / 255.0 * 2.0 * 3.14159;

	if (result > 2.0 * 3.141592) {
		result = result - 2.0 * 3.141592;
	}
	if (result < 0) {
		result = result + 2.0 * 3.141592;
	}
	return result / 2.0 / 3.141592;
}

// convert float network output to unchar image 
__global__ void networkToImageKernel(unsigned char* imageOutput, float* networkOutput, int W, int H, int C) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= W || v >= H)
		return;
	int imageIndex = (u + v*W) * C;
	imageOutput[imageIndex] = (unsigned char)floor(phaseToImage(networkOutput[(0 * H + v) * W + u],185.0f) * 255.0); //B
	imageOutput[imageIndex + 1] = (unsigned char)floor(phaseToImage(networkOutput[(1 * H + v) * W + u],230.0f) * 255.0); //G
	imageOutput[imageIndex + 2] = (unsigned char)floor(phaseToImage(networkOutput[(2 * H + v) * W + u] , 125.0f) * 255.0); //R
}

// convert to opengl format
__global__ void imageToOpenglCudaKernel(unsigned char* imageOutput, uchar4* openglDevice, int W, int H, int C) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= W || v >= H)
		return;
	int ptrGL = u + v * W;
	int ptrImage = (u + v * W) * C;

	openglDevice[ptrGL].z = imageOutput[ptrImage];
	openglDevice[ptrGL].y = imageOutput[ptrImage + 1];
	openglDevice[ptrGL].x = imageOutput[ptrImage + 2];
	openglDevice[ptrGL].w = 0;
	//openglDevice[ptrGL].z = 255;
	//openglDevice[ptrGL].y = 255;
	//openglDevice[ptrGL].x = 255;
	//openglDevice[ptrGL].w = 0;
	//bgr -> rgb
}

// fill
__global__ void fillCudaKernel(float* inputDevice,  int W, int H, int C) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= W || v >= H)
		return;
	int index = (u + v * W) * C;
	inputDevice[index] = 0.0;
}

extern "C"  void nothingCuda(unsigned char* outputDevice, unsigned char* inputDevice, int W, int H, int C) {
	dim3 block(32, 32);
	dim3 grid((W * 2 + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	nothing << <grid, block >> > (outputDevice, inputDevice, W, H, C);
}
extern "C"  void calMapCuda(float2 * mapDevice, int W, int H, int C, int cutW, int cutH, int shift) {
	dim3 block(24, 24);
	dim3 grid((cutW * 2 + block.x - 1) / block.x, (cutH*2 + block.y - 1) / block.y);
	calMapCudaKernel << <grid, block >> > (mapDevice, W, H, C, cutW, cutH, shift);
}

extern "C"  void applyMapCuda(unsigned char* encodeDevice, unsigned char* inputDevice1,
	unsigned char* inputDevice2, unsigned char* inputDevice3, unsigned char* inputDevice4,
	float2 * mapDevice, int W, int H, int C, int cutW, int cutH) {
	dim3 block(32, 32);
	dim3 grid((cutW * 2 + block.x - 1) / block.x, (cutH*2 + block.y - 1) / block.y);
	applyMapCudaKernel << <grid, block >> > (encodeDevice, inputDevice1,
		inputDevice2, inputDevice3, inputDevice4, mapDevice, W, H, C, cutW, cutH);
}
// (W,H)*4 images to (W *2,H*2) encoded image
extern "C"  void encodeImageCuda(unsigned char* encodeDevice, unsigned char* inputDevice1,
	unsigned char* inputDevice2, unsigned char* inputDevice3, unsigned char* inputDevice4,
	int W, int H, int C, int cutW, int cutH, int shift)  {
	dim3 block(32, 32);
	dim3 grid((cutW * 2 + block.x - 1) / block.x, (cutH*2 + block.y - 1) / block.y);
	encodeImageCudaKernel << <grid, block >> > (encodeDevice, inputDevice1,inputDevice2, 
		inputDevice3, inputDevice4, W, H, C, cutW, cutH, shift);
}

extern "C"  void imageToNetworkCuda(unsigned char* imageInput, float* networkInput, int W, int H, int C) {
	dim3 block(32, 32);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	imageToNetworkKernel << <grid, block >> > (imageInput, networkInput, W, H, C);
}

extern "C"  void networkToImageCuda(unsigned char* imageOutput, float* networkOutput, int W, int H, int C) {
	dim3 block(24, 24);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	networkToImageKernel << <grid, block >> > (imageOutput, networkOutput, W, H, C);
}

extern "C"  void imageToOpenglCuda(unsigned char* imageOutput, uchar4* openglDevice, int W, int H, int C) {
	dim3 block(24, 24);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	imageToOpenglCudaKernel << <grid, block >> > (imageOutput, openglDevice, W, H, C);
}

extern "C"  void undistortCuda(unsigned char* inputDevice, unsigned char* outputDevice, int W, int H, int C){
	dim3 block(24, 24);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	undistortCudaKernel << <grid, block >> > (inputDevice, outputDevice, W, H, C);
}

extern "C"  void homographyCuda(unsigned char* inputDevice, unsigned char* outputDevice, int W, int H, int C, int index){
	dim3 block(32, 32);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	homographyCudaKernel << <grid, block >> > (inputDevice, outputDevice, W, H, C, index);
}

extern "C"  void fillCuda(float* inputDevice, int W, int H, int C) {
	dim3 block(32, 32);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	fillCudaKernel << <grid, block >> > (inputDevice,  W, H, C);
}
