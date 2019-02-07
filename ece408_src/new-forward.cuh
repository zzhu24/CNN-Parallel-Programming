#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 32

#define SHARED_SIZE 38

#define MASK_SIZE 7

#define COALESCENCE_BLOCK 4

#define LARGE_OUTPUT 24

#define LARGE_INPUT 12

#define SMALL_OUTPUT 12

#define SMALL_INPUT 1

#define MAX_THREADS 256


#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


__constant__ float LARGE_MASK[LARGE_OUTPUT * LARGE_INPUT * MASK_SIZE * MASK_SIZE];
__constant__ float SMALL_MASK[SMALL_OUTPUT * SMALL_INPUT * MASK_SIZE * MASK_SIZE];







__global__ void forward_kernel_original(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

	/*
	Modify this function to implement the forward pass described in Chapter 16.
	We have added an additional dimension to the tensors to support an entire mini-batch
	The goal here is to be correct AND fast.
	We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	*/

	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define mask4d(i3, i2, i1, i0) LARGE_MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));


	int n = blockIdx.x * 4;
	// int n = blockIdx.x;
	int m = blockIdx.y * 4;
	// int m = blockIdx.y;


	int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
	int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;


	float acc0 = 0.0;
	float acc1 = 0.0;
	float acc2 = 0.0;
	float acc3 = 0.0;
	float acc4 = 0.0;
	float acc5 = 0.0;
	float acc6 = 0.0;
	float acc7 = 0.0;
	float acc8 = 0.0;
	float acc9 = 0.0;
	float acc10 = 0.0;
	float acc11 = 0.0;
	float acc12 = 0.0;
	float acc13 = 0.0;
	float acc14 = 0.0;
	float acc15 = 0.0;

	if (n < B && m < M && h < H_out && w < W_out){

		for( int c = 0; c < C; c++ ){
			for( int p = 0; p < K; p++ ){
				for( int q = 0; q < K; q++ ){
					acc0 += x4d(n, c,  h + p, w + q) * mask4d(m, c, p, q);
					acc1 += x4d(n , c,  h + p, w + q) * mask4d(m+1 , c, p, q);
					acc2 += x4d(n , c,  h + p, w + q) * mask4d(m+2 , c, p, q);
					acc3 += x4d(n , c,  h + p, w + q) * mask4d(m+3, c, p, q);
					acc4 += x4d(n+1, c,  h + p, w + q) * mask4d(m, c, p, q);
					acc5 += x4d(n+1 , c,  h + p, w + q) * mask4d(m+1 , c, p, q);
					acc6 += x4d(n+1 , c,  h + p, w + q) * mask4d(m+2 , c, p, q);
					acc7 += x4d(n+1 , c,  h + p, w + q) * mask4d(m+3, c, p, q);
					acc8 += x4d(n+2, c,  h + p, w + q) * mask4d(m, c, p, q);
					acc9 += x4d(n+2 , c,  h + p, w + q) * mask4d(m+1 , c, p, q);
					acc10 += x4d(n+2 , c,  h + p, w + q) * mask4d(m+2 , c, p, q);
					acc11 += x4d(n+2 , c,  h + p, w + q) * mask4d(m+3, c, p, q);
					acc12 += x4d(n+3, c,  h + p, w + q) * mask4d(m, c, p, q);
					acc13 += x4d(n+3 , c,  h + p, w + q) * mask4d(m+1 , c, p, q);
					acc14 += x4d(n+3 , c,  h + p, w + q) * mask4d(m+2 , c, p, q);
					acc15 += x4d(n+3 , c,  h + p, w + q) * mask4d(m+3, c, p, q);
				}
			}
		}


		y4d(n, m, h, w) = acc0;
		y4d(n, m+1, h, w) = acc1;
		y4d(n, m+2, h, w) = acc2;
		y4d(n, m+3, h, w) = acc3;
		y4d(n+1, m, h, w) = acc4;
		y4d(n+1, m+1, h, w) = acc5;
		y4d(n+1, m+2, h, w) = acc6;
		y4d(n+1, m+3, h, w) = acc7;
		y4d(n+2, m, h, w) = acc8;
		y4d(n+2, m+1, h, w) = acc9;
		y4d(n+2, m+2, h, w) = acc10;
		y4d(n+2, m+3, h, w) = acc11;
		y4d(n+3, m, h, w) = acc12;
		y4d(n+3, m+1, h, w) = acc13;
		y4d(n+3, m+2, h, w) = acc14;
		y4d(n+3, m+3, h, w) = acc15;
	}


	#undef y4d
	#undef x4d
	#undef mask4d
}







__global__ void forward_kernel_first(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

	/*
	Modify this function to implement the forward pass described in Chapter 16.
	We have added an additional dimension to the tensors to support an entire mini-batch
	The goal here is to be correct AND fast.
	We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	*/

	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define mask4d(i2, i1, i0) SMALL_MASK[(i2) * (MASK_SIZE * MASK_SIZE) + (i1) * (MASK_SIZE) + i0]

	int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));


	int n = blockIdx.x * 4;
	// int n = blockIdx.x;
	int m = blockIdx.y * 4;
	// int m = blockIdx.y;


	int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
	int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;

	//int h0 = threadIdx.y;
 	//int w0 = threadIdx.x;

	//int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
	//int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;

	//printf("B: %d\nM: %d\nC: %d\nH: %d\nW: %d\nK: %d\n", B, M, C, H, W, K);


	//__shared__ float X_Shared[SHARED_SIZE * SHARED_SIZE];

	//printf("B: %d\nM: %d\nC: %d\nH: %d\nW: %d\nK: %d\n", B, M, C, H, W, K);


	/*
	#define shared4d(i1, i0) X_Shared[(i1) * (SHARED_SIZE) + i0]

	for(int i = h; i < h_base + SHARED_SIZE; i += TILE_WIDTH){
		for(int j = w; j < w_base + SHARED_SIZE; j += TILE_WIDTH){
			if(i < H && j < W){
				//printf("Assigning Values !\n");
				shared4d(i - h_base, j - w_base) = x4d(n, 0, i, j);
			}
			else{
				//printf("Assigning Zeros !\n");
				shared4d(i - h_base, j - w_base) = 0.0;
			}
		}
	}

 	__syncthreads();
 	*/


	float acc0 = 0.0;
	float acc1 = 0.0;
	float acc2 = 0.0;
	float acc3 = 0.0;
	float acc4 = 0.0;
	float acc5 = 0.0;
	float acc6 = 0.0;
	float acc7 = 0.0;
	float acc8 = 0.0;
	float acc9 = 0.0;
	float acc10 = 0.0;
	float acc11 = 0.0;
	float acc12 = 0.0;
	float acc13 = 0.0;
	float acc14 = 0.0;
	float acc15 = 0.0;

	if (n < B && m < M && h < H_out && w < W_out){

		for( int p = 0; p < K; p++ ){
			for( int q = 0; q < K; q++ ){
				acc0 += x4d(n, 0,  h + p, w + q) * mask4d(m, p, q);
				acc1 += x4d(n , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc2 += x4d(n , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc3 += x4d(n , 0,  h + p, w + q) * mask4d(m+3, p, q);
				acc4 += x4d(n+1, 0,  h + p, w + q) * mask4d(m, p, q);
				acc5 += x4d(n+1 , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc6 += x4d(n+1 , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc7 += x4d(n+1 , 0,  h + p, w + q) * mask4d(m+3, p, q);
				acc8 += x4d(n+2, 0,  h + p, w + q) * mask4d(m, p, q);
				acc9 += x4d(n+2 , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc10 += x4d(n+2 , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc11 += x4d(n+2 , 0,  h + p, w + q) * mask4d(m+3, p, q);
				acc12 += x4d(n+3, 0,  h + p, w + q) * mask4d(m, p, q);
				acc13 += x4d(n+3 , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc14 += x4d(n+3 , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc15 += x4d(n+3 , 0,  h + p, w + q) * mask4d(m+3, p, q);
			}
		}


		y4d(n, m, h, w) = acc0;
		y4d(n, m+1, h, w) = acc1;
		y4d(n, m+2, h, w) = acc2;
		y4d(n, m+3, h, w) = acc3;
		y4d(n+1, m, h, w) = acc4;
		y4d(n+1, m+1, h, w) = acc5;
		y4d(n+1, m+2, h, w) = acc6;
		y4d(n+1, m+3, h, w) = acc7;
		y4d(n+2, m, h, w) = acc8;
		y4d(n+2, m+1, h, w) = acc9;
		y4d(n+2, m+2, h, w) = acc10;
		y4d(n+2, m+3, h, w) = acc11;
		y4d(n+3, m, h, w) = acc12;
		y4d(n+3, m+1, h, w) = acc13;
		y4d(n+3, m+2, h, w) = acc14;
		y4d(n+3, m+3, h, w) = acc15;

	}


	#undef y4d
	#undef x4d
	#undef mask4d
	#undef shared4d
}



__global__ void forward_kernel_second(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

	/*
	Modify this function to implement the forward pass described in Chapter 16.
	We have added an additional dimension to the tensors to support an entire mini-batch
	The goal here is to be correct AND fast.
	We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	*/

	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define mask4d(i3, i2, i1, i0) LARGE_MASK[(i3) * (C * MASK_SIZE * MASK_SIZE) + (i2) * (MASK_SIZE * MASK_SIZE) + (i1) * (MASK_SIZE) + i0]

	int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));


	int n = blockIdx.x * 4;
	// int n = blockIdx.x;
	int m = blockIdx.y * 4;
	// int m = blockIdx.y;


	int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
	int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;

	int h0 = threadIdx.y;
 	int w0 = threadIdx.x;

	int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
	int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;


	__shared__ float X_Shared[COALESCENCE_BLOCK * (SHARED_SIZE * SHARED_SIZE)];


	#define shared4d(i2, i1, i0) X_Shared[(i2) * (SHARED_SIZE * SHARED_SIZE) + (i1) * (SHARED_SIZE) + i0]



	float acc0 = 0.0;
	float acc1 = 0.0;
	float acc2 = 0.0;
	float acc3 = 0.0;
	float acc4 = 0.0;
	float acc5 = 0.0;
	float acc6 = 0.0;
	float acc7 = 0.0;
	float acc8 = 0.0;
	float acc9 = 0.0;
	float acc10 = 0.0;
	float acc11 = 0.0;
	float acc12 = 0.0;
	float acc13 = 0.0;
	float acc14 = 0.0;
	float acc15 = 0.0;

	for(int c = 0; c < C; c++){

		for(int i = h; i < h_base + SHARED_SIZE; i += TILE_WIDTH){
			for(int j = w; j < w_base + SHARED_SIZE; j += TILE_WIDTH){
				if(i < H && j < W){
					shared4d(0, i - h_base, j - w_base) = x4d(n, c, i, j);
					shared4d(1, i - h_base, j - w_base) = x4d(n+1, c, i, j);
					shared4d(2, i - h_base, j - w_base) = x4d(n+2, c, i, j);
					shared4d(3, i - h_base, j - w_base) = x4d(n+3, c, i, j);
				}
				else{
					
					shared4d(0, i - h_base, j - w_base) = 0.0;
					shared4d(1, i - h_base, j - w_base) = 0.0;
					shared4d(2, i - h_base, j - w_base) = 0.0;
					shared4d(3, i - h_base, j - w_base) = 0.0;
				}
			}
		}

	 	__syncthreads();

	 	for( int p = 0; p < K; p++ ){
			for( int q = 0; q < K; q++ ){
				acc0 += shared4d(0,  h0 + p, w0 + q) * mask4d(m, c, p, q);
				acc1 += shared4d(0,  h0 + p, w0 + q) * mask4d(m+1 , c, p, q);
				acc2 += shared4d(0,  h0 + p, w0 + q) * mask4d(m+2 , c, p, q);
				acc3 += shared4d(0,  h0 + p, w0 + q) * mask4d(m+3, c, p, q);
				acc4 += shared4d(1,  h0 + p, w0 + q) * mask4d(m, c, p, q);
				acc5 += shared4d(1,  h0 + p, w0 + q) * mask4d(m+1 , c, p, q);
				acc6 += shared4d(1,  h0 + p, w0 + q) * mask4d(m+2 , c, p, q);
				acc7 += shared4d(1,  h0 + p, w0 + q) * mask4d(m+3, c, p, q);
				acc8 += shared4d(2,  h0 + p, w0 + q) * mask4d(m, c, p, q);
				acc9 += shared4d(2,  h0 + p, w0 + q) * mask4d(m+1 , c, p, q);
				acc10 += shared4d(2,  h0 + p, w0 + q) * mask4d(m+2 , c, p, q);
				acc11 += shared4d(2,  h0 + p, w0 + q) * mask4d(m+3, c, p, q);
				acc12 += shared4d(3,  h0 + p, w0 + q) * mask4d(m, c, p, q);
				acc13 += shared4d(3,  h0 + p, w0 + q) * mask4d(m+1 , c, p, q);
				acc14 += shared4d(3,  h0 + p, w0 + q) * mask4d(m+2 , c, p, q);
				acc15 += shared4d(3,  h0 + p, w0 + q) * mask4d(m+3, c, p, q);
				
			}
		}
		

		//__syncthreads();

			 
	}


	y4d(n, m, h, w) = acc0;
	y4d(n, m+1, h, w) = acc1;
	y4d(n, m+2, h, w) = acc2;
	y4d(n, m+3, h, w) = acc3;
	y4d(n+1, m, h, w) = acc4;
	y4d(n+1, m+1, h, w) = acc5;
	y4d(n+1, m+2, h, w) = acc6;
	y4d(n+1, m+3, h, w) = acc7;
	y4d(n+2, m, h, w) = acc8;
	y4d(n+2, m+1, h, w) = acc9;
	y4d(n+2, m+2, h, w) = acc10;
	y4d(n+2, m+3, h, w) = acc11;
	y4d(n+3, m, h, w) = acc12;
	y4d(n+3, m+1, h, w) = acc13;
	y4d(n+3, m+2, h, w) = acc14;
	y4d(n+3, m+3, h, w) = acc15;







	#undef y4d
	#undef x4d
	#undef mask4d
	#undef shared4d
}








__global__ void forward_kernel_atomic(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

	/*
	Modify this function to implement the forward pass described in Chapter 16.
	We have added an additional dimension to the tensors to support an entire mini-batch
	The goal here is to be correct AND fast.
	We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	*/

	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define mask4d(i2, i1, i0) SMALL_MASK[(i2) * (MASK_SIZE * MASK_SIZE) + (i1) * (MASK_SIZE) + i0]

	int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));


	int n = blockIdx.x * 4;
	// int n = blockIdx.x;
	int m = blockIdx.y * 4;
	// int m = blockIdx.y;


	int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
	int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;

	//int h0 = threadIdx.y;
 	//int w0 = threadIdx.x;

	//int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
	//int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;

	//printf("B: %d\nM: %d\nC: %d\nH: %d\nW: %d\nK: %d\n", B, M, C, H, W, K);


	//__shared__ float X_Shared[SHARED_SIZE * SHARED_SIZE];

	//printf("B: %d\nM: %d\nC: %d\nH: %d\nW: %d\nK: %d\n", B, M, C, H, W, K);


	/*
	#define shared4d(i1, i0) X_Shared[(i1) * (SHARED_SIZE) + i0]

	for(int i = h; i < h_base + SHARED_SIZE; i += TILE_WIDTH){
		for(int j = w; j < w_base + SHARED_SIZE; j += TILE_WIDTH){
			if(i < H && j < W){
				//printf("Assigning Values !\n");
				shared4d(i - h_base, j - w_base) = x4d(n, 0, i, j);
			}
			else{
				//printf("Assigning Zeros !\n");
				shared4d(i - h_base, j - w_base) = 0.0;
			}
		}
	}

 	__syncthreads();
 	*/


	float acc0 = 0.0;
	float acc1 = 0.0;
	float acc2 = 0.0;
	float acc3 = 0.0;
	float acc4 = 0.0;
	float acc5 = 0.0;
	float acc6 = 0.0;
	float acc7 = 0.0;
	float acc8 = 0.0;
	float acc9 = 0.0;
	float acc10 = 0.0;
	float acc11 = 0.0;
	float acc12 = 0.0;
	float acc13 = 0.0;
	float acc14 = 0.0;
	float acc15 = 0.0;

	if (n < B && m < M && h < H_out && w < W_out){

		for( int p = 0; p < K; p++ ){
			for( int q = 0; q < K; q++ ){
				acc0 += x4d(n, 0,  h + p, w + q) * mask4d(m, p, q);
				acc1 += x4d(n , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc2 += x4d(n , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc3 += x4d(n , 0,  h + p, w + q) * mask4d(m+3, p, q);
				acc4 += x4d(n+1, 0,  h + p, w + q) * mask4d(m, p, q);
				acc5 += x4d(n+1 , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc6 += x4d(n+1 , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc7 += x4d(n+1 , 0,  h + p, w + q) * mask4d(m+3, p, q);
				acc8 += x4d(n+2, 0,  h + p, w + q) * mask4d(m, p, q);
				acc9 += x4d(n+2 , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc10 += x4d(n+2 , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc11 += x4d(n+2 , 0,  h + p, w + q) * mask4d(m+3, p, q);
				acc12 += x4d(n+3, 0,  h + p, w + q) * mask4d(m, p, q);
				acc13 += x4d(n+3 , 0,  h + p, w + q) * mask4d(m+1, p, q);
				acc14 += x4d(n+3 , 0,  h + p, w + q) * mask4d(m+2, p, q);
				acc15 += x4d(n+3 , 0,  h + p, w + q) * mask4d(m+3, p, q);
			}
		}

    atomicAdd(&y4d(n,m,h,w),acc0);
    atomicAdd(&y4d(n,m+1,h,w),acc1);
    atomicAdd(&y4d(n,m+2,h,w),acc2);
    atomicAdd(&y4d(n,m+3,h,w),acc3);
    atomicAdd(&y4d(n+1,m,h,w),acc4);
    atomicAdd(&y4d(n+1,m+1,h,w),acc5);
    atomicAdd(&y4d(n+1,m+2,h,w),acc6);
    atomicAdd(&y4d(n+1,m+3,h,w),acc7);
    atomicAdd(&y4d(n+2,m,h,w),acc8);
    atomicAdd(&y4d(n+2,m+1,h,w),acc9);
    atomicAdd(&y4d(n+2,m+2,h,w),acc10);
    atomicAdd(&y4d(n+2,m+3,h,w),acc11);
    atomicAdd(&y4d(n+3,m,h,w),acc12);
    atomicAdd(&y4d(n+3,m+1,h,w),acc13);
    atomicAdd(&y4d(n+3,m+2,h,w),acc14);
    atomicAdd(&y4d(n+3,m+3,h,w),acc15);


	/*	y4d(n, m, h, w) = acc0;
		y4d(n, m+1, h, w) = acc1;
		y4d(n, m+2, h, w) = acc2;
		y4d(n, m+3, h, w) = acc3;
		y4d(n+1, m, h, w) = acc4;
		y4d(n+1, m+1, h, w) = acc5;
		y4d(n+1, m+2, h, w) = acc6;
		y4d(n+1, m+3, h, w) = acc7;
		y4d(n+2, m, h, w) = acc8;
		y4d(n+2, m+1, h, w) = acc9;
		y4d(n+2, m+2, h, w) = acc10;
		y4d(n+2, m+3, h, w) = acc11;
		y4d(n+3, m, h, w) = acc12;
		y4d(n+3, m+1, h, w) = acc13;
		y4d(n+3, m+2, h, w) = acc14;
		y4d(n+3, m+3, h, w) = acc15; */

	}


	#undef y4d
	#undef x4d
	#undef mask4d
	#undef shared4d
}



__global__ void forward_kernel_loop(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //printf("B: %d\nM: %d\nC: %d\nH: %d\nW: %d\nK: %d\n", B, M, C, H, W, K);
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask4d(i3, i2, i1, i0) LARGE_MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));


    int n = blockIdx.x * 2;
    // int n = blockIdx.x;
    int m = blockIdx.y * 2;
    // int m = blockIdx.y;

    int h0 = threadIdx.y;
    int w0 = threadIdx.x;

    int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + h0;
    int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + w0;



 if (n < B && m < M && h < H_out && w < W_out){
        y4d(n, m, h, w) = 0.0;
        y4d(n, m+1, h, w) = 0.0;
        y4d(n+1, m+1, h, w) = 0.0;
        y4d(n+1, m, h, w) = 0.0;
        for( int c = 0; c < C; c++ ){
                    //y4d(n, m, h, w) += x4d(n, c,  h + p, w + q) * mask4d(m, c, p, q);
                    //y4d(n, m+1, h, w) += x4d(n, c,  h + p, w + q) * mask4d(m + 1, c, p, q);
                    //y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + p, w + q) * mask4d(m + 1, c, p, q);
                    //y4d(n+1, m, h, w) += x4d(n + 1, c,  h + p, w + q) * mask4d(m, c, p, q);

                    y4d(n, m, h, w) += x4d(n, c,  h, w) * mask4d(m, c, 0, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h, w ) * mask4d(m + 1, c, 0, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h, w ) * mask4d(m + 1, c, 0, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h, w ) * mask4d(m, c, 0, 0);

                    y4d(n, m, h, w) += x4d(n, c,  h + 0, w + 1) * mask4d(m, c, 0, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 0, w + 1) * mask4d(m + 1, c, 0, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 0, w + 1) * mask4d(m + 1, c, 0, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 0, w + 1) * mask4d(m, c, 0, 1);

                    y4d(n, m, h, w) += x4d(n, c,  h + 0, w + 2) * mask4d(m, c, 0, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 0, w + 2) * mask4d(m + 1, c, 0, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 0, w + 2) * mask4d(m + 1, c, 0, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 0, w + 2) * mask4d(m, c, 0, 2);


                    y4d(n, m, h, w) += x4d(n, c,  h + 0, w + 3) * mask4d(m, c, 0, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 0, w + 3) * mask4d(m + 1, c, 0, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 0, w + 3) * mask4d(m + 1, c, 0, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 0, w + 3) * mask4d(m, c, 0, 3);


                    y4d(n, m, h, w) += x4d(n, c,  h + 0, w + 4) * mask4d(m, c, 0, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 0, w + 4) * mask4d(m + 1, c, 0, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 0, w + 4) * mask4d(m + 1, c, 0, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 0, w + 4) * mask4d(m, c, 0, 4);


                    y4d(n, m, h, w) += x4d(n, c,  h + 0, w + 5) * mask4d(m, c, 0, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 0, w + 5) * mask4d(m + 1, c, 0, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 0, w + 5) * mask4d(m + 1, c, 0, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 0, w + 5) * mask4d(m, c, 0, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 0, w + 6) * mask4d(m, c, 0, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 0, w + 6) * mask4d(m + 1, c, 0, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 0, w + 6) * mask4d(m + 1, c, 0, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 0, w + 6) * mask4d(m, c, 0, 6);


                    y4d(n, m, h, w) += x4d(n, c,  h+1, w) * mask4d(m, c, 1, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h+1, w ) * mask4d(m + 1, c, 1, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h+1, w ) * mask4d(m + 1, c, 1, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h+1, w ) * mask4d(m, c, 1, 0);

                    y4d(n, m, h, w) += x4d(n, c,  h + 1, w + 1) * mask4d(m, c, 1, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 1, w + 1) * mask4d(m + 1, c, 1, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 1, w + 1) * mask4d(m + 1, c, 1, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 1, w + 1) * mask4d(m, c, 1, 1);

                    y4d(n, m, h, w) += x4d(n, c,  h + 1, w + 2) * mask4d(m, c, 1, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 1, w + 2) * mask4d(m + 1, c, 1, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 1, w + 2) * mask4d(m + 1, c, 1, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 1, w + 2) * mask4d(m, c, 1, 2);


                    y4d(n, m, h, w) += x4d(n, c,  h + 1, w + 3) * mask4d(m, c, 1, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 1, w + 3) * mask4d(m + 1, c, 1, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 1, w + 3) * mask4d(m + 1, c, 1, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 1, w + 3) * mask4d(m, c, 1, 3);


                    y4d(n, m, h, w) += x4d(n, c,  h + 1, w + 4) * mask4d(m, c, 1, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 1, w + 4) * mask4d(m + 1, c, 1, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 1, w + 4) * mask4d(m + 1, c, 1, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 1, w + 4) * mask4d(m, c, 1, 4);


                    y4d(n, m, h, w) += x4d(n, c,  h + 1, w + 5) * mask4d(m, c, 1, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 1, w + 5) * mask4d(m + 1, c, 1, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 1, w + 5) * mask4d(m + 1, c, 1, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 1, w + 5) * mask4d(m, c, 1, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 1, w + 6) * mask4d(m, c, 1, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 1, w + 6) * mask4d(m + 1, c, 1, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 1, w + 6) * mask4d(m + 1, c, 1, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 1, w + 6) * mask4d(m, c, 1, 6);

                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 0) * mask4d(m, c, 2, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 0) * mask4d(m + 1, c, 2, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 0) * mask4d(m + 1, c, 2, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 0) * mask4d(m, c, 2, 0);

                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 1) * mask4d(m, c, 2, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 1) * mask4d(m + 1, c, 2, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 1) * mask4d(m + 1, c, 2, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 1) * mask4d(m, c, 2, 1);


                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 2) * mask4d(m, c, 2, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 2) * mask4d(m + 1, c, 2, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 2) * mask4d(m + 1, c, 2, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 2) * mask4d(m, c, 2, 2);


                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 3) * mask4d(m, c, 2, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 3) * mask4d(m + 1, c, 2, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 3) * mask4d(m + 1, c, 2, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 3) * mask4d(m, c, 2, 3);


                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 4) * mask4d(m, c, 2, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 4) * mask4d(m + 1, c, 2, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 4) * mask4d(m + 1, c, 2, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 4) * mask4d(m, c, 2, 4);


                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 5) * mask4d(m, c, 2, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 5) * mask4d(m + 1, c, 2, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 5) * mask4d(m + 1, c, 2, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 5) * mask4d(m, c, 2, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 2, w + 6) * mask4d(m, c, 2, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 2, w + 6) * mask4d(m + 1, c, 2, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 2, w + 6) * mask4d(m + 1, c, 2, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 2, w + 6) * mask4d(m, c, 2, 6);

                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 0) * mask4d(m, c, 3, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 0) * mask4d(m + 1, c, 3, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 0) * mask4d(m + 1, c, 3, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 0) * mask4d(m, c, 3, 0);


                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 1) * mask4d(m, c, 3, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 1) * mask4d(m + 1, c, 3, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 1) * mask4d(m + 1, c, 3, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 1) * mask4d(m, c, 3, 1);


                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 2) * mask4d(m, c, 3, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 2) * mask4d(m + 1, c, 3, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 2) * mask4d(m + 1, c, 3, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 2) * mask4d(m, c, 3, 2);


                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 3) * mask4d(m, c, 3, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 3) * mask4d(m + 1, c, 3, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 3) * mask4d(m + 1, c, 3, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 3) * mask4d(m, c, 3, 3);


                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 4) * mask4d(m, c, 3, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 4) * mask4d(m + 1, c, 3, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 4) * mask4d(m + 1, c, 3, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 4) * mask4d(m, c, 3, 4);


                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 5) * mask4d(m, c, 3, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 5) * mask4d(m + 1, c, 3, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 5) * mask4d(m + 1, c, 3, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 5) * mask4d(m, c, 3, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 3, w + 6) * mask4d(m, c, 3, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 3, w + 6) * mask4d(m + 1, c, 3, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 3, w + 6) * mask4d(m + 1, c, 3, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 3, w + 6) * mask4d(m, c, 3, 6);

                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 0) * mask4d(m, c, 4, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 0) * mask4d(m + 1, c, 4, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 0) * mask4d(m + 1, c, 4, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 0) * mask4d(m, c, 4, 0);


                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 1) * mask4d(m, c, 4, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 1) * mask4d(m + 1, c, 4, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 1) * mask4d(m + 1, c, 4, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 1) * mask4d(m, c, 4, 1);


                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 2) * mask4d(m, c, 4, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 2) * mask4d(m + 1, c, 4, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 2) * mask4d(m + 1, c, 4, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 2) * mask4d(m, c, 4, 2);


                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 3) * mask4d(m, c, 4, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 3) * mask4d(m + 1, c, 4, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 3) * mask4d(m + 1, c, 4, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 3) * mask4d(m, c, 4, 3);


                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 4) * mask4d(m, c, 4, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 4) * mask4d(m + 1, c, 4, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 4) * mask4d(m + 1, c, 4, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 4) * mask4d(m, c, 4, 4);


                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 5) * mask4d(m, c, 4, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 5) * mask4d(m + 1, c, 4, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 5) * mask4d(m + 1, c, 4, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 5) * mask4d(m, c, 4, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 4, w + 6) * mask4d(m, c, 4, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 4, w + 6) * mask4d(m + 1, c, 4, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 4, w + 6) * mask4d(m + 1, c, 4, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 4, w + 6) * mask4d(m, c, 4, 6);

                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 0) * mask4d(m, c, 5, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 0) * mask4d(m + 1, c, 5, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 0) * mask4d(m + 1, c, 5, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 0) * mask4d(m, c, 5, 0);


                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 1) * mask4d(m, c, 5, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 1) * mask4d(m + 1, c, 5, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 1) * mask4d(m + 1, c, 5, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 1) * mask4d(m, c, 5, 1);

                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 2) * mask4d(m, c, 5, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 2) * mask4d(m + 1, c, 5, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 2) * mask4d(m + 1, c, 5, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 2) * mask4d(m, c, 5, 2);

                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 3) * mask4d(m, c, 5, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 3) * mask4d(m + 1, c, 5, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 3) * mask4d(m + 1, c, 5, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 3) * mask4d(m, c, 5, 3);

                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 4) * mask4d(m, c, 5, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 4) * mask4d(m + 1, c, 5, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 4) * mask4d(m + 1, c, 5, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 4) * mask4d(m, c, 5, 4);

                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 5) * mask4d(m, c, 5, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 5) * mask4d(m + 1, c, 5, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 5) * mask4d(m + 1, c, 5, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 5) * mask4d(m, c, 5, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 5, w + 6) * mask4d(m, c, 5, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 5, w + 6) * mask4d(m + 1, c, 5, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 5, w + 6) * mask4d(m + 1, c, 5, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 5, w + 6) * mask4d(m, c, 5, 6);

                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 0) * mask4d(m, c, 6, 0);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 0) * mask4d(m + 1, c, 6, 0);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 0) * mask4d(m + 1, c, 6, 0);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 0) * mask4d(m, c, 6, 0);


                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 1) * mask4d(m, c, 6, 1);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 1) * mask4d(m + 1, c, 6, 1);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 1) * mask4d(m + 1, c, 6, 1);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 1) * mask4d(m, c, 6, 1);


                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 2) * mask4d(m, c, 6, 2);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 2) * mask4d(m + 1, c, 6, 2);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 2) * mask4d(m + 1, c, 6, 2);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 2) * mask4d(m, c, 6, 2);


                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 3) * mask4d(m, c, 6, 3);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 3) * mask4d(m + 1, c, 6, 3);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 3) * mask4d(m + 1, c, 6, 3);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 3) * mask4d(m, c, 6, 3);


                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 4) * mask4d(m, c, 6, 4);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 4) * mask4d(m + 1, c, 6, 4);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 4) * mask4d(m + 1, c, 6, 4);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 4) * mask4d(m, c, 6, 4);


                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 5) * mask4d(m, c, 6, 5);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 5) * mask4d(m + 1, c, 6, 5);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 5) * mask4d(m + 1, c, 6, 5);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 5) * mask4d(m, c, 6, 5);


                    y4d(n, m, h, w) += x4d(n, c,  h + 6, w + 6) * mask4d(m, c, 6, 6);
                    y4d(n, m+1, h, w) += x4d(n, c,  h + 6, w + 6) * mask4d(m + 1, c, 6, 6);
                    y4d(n+1, m+1, h, w) += x4d(n + 1, c,  h + 6, w + 6) * mask4d(m + 1, c, 6, 6);
                    y4d(n+1, m, h, w) += x4d(n + 1, c,  h + 6, w + 6) * mask4d(m, c, 6, 6);


}

}


#undef y4d
#undef x4d
#undef mask4d
}






__global__ void unroll_Kernel(int C, int H, int W, int K, float* x,  float* xout) {
  int t = blockIdx.x * MAX_THREADS + threadIdx.x;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int W_unroll = H_out * W_out;
  
  #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + (i0)]
  if (t < C*W_unroll) {
    int c = t / W_unroll;
    int s = t % W_unroll;
    int h_out = s/W_out;
    int w_out = s%W_out;
    int h_unroll = h_out * W_out + w_out;
    int w_base = c*K*K;
    for(int p = 0; p < K; p++) {
      for(int q = 0; q < K; q++) {
        int w_unroll = w_base + p * K + q;
        xout[(w_unroll)*W_unroll + h_unroll] = x3d(c,(h_out + p),(w_out + q)); 
      }
    }
  }
  #undef x3d
}



/*
 This function is called by new-inl.h
 Any code you writ e should be executed by this function.
 For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

	// Use mxnet's CHECK_EQ to do assertions.
	// Remove this assertion when you do your implementation!


	// Extract the tensor dimensions into B,M,C,H,W,K
	// ...


	const int B = x.shape_[0];
	const int M = y.shape_[1];
	const int C = x.shape_[1];
	const int H = x.shape_[2];
	const int W = x.shape_[3];
	const int K = w.shape_[3];

	const int H_out = H - K + 1;
	const int W_out = W - K + 1;
	int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
	int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
	const int Z = H_grid * W_grid;

	// Set the kernel dimensions

	dim3 gridDimSimple(ceil(B / 2.0), ceil(M / 2.0), Z);
	dim3 gridDim(ceil(B / COALESCENCE_BLOCK * 1.0), ceil(M / COALESCENCE_BLOCK * 1.0), Z);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	//cudaMemcpyToSymbol(LARGE_MASK, w.dptr_, C * M * K * K * sizeof(float));
	//forward_kernel_loop<<<gridDimSimple, blockDim>>>(y.dptr_,x.dptr_, B, M, C, H, W, K);

	//float * xout;
	//cudaMalloc((void**) &xout, sizeof(float) * H_out * W_out * (C*K*K));
	//unroll_Kernel<<<blockDim, MAX_THREADS>>> (C, H, W, K, x.dptr_, xout);
	//forward_kernel_atomic<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B, M, C, H, W, K)


	if(C == 1){
		cudaMemcpyToSymbol(SMALL_MASK, w.dptr_, C * M * K * K * sizeof(float));
		forward_kernel_first<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B, M, C, H, W, K);
	}
	else if(C == 12){
		cudaMemcpyToSymbol(LARGE_MASK, w.dptr_, C * M * K * K * sizeof(float));
		forward_kernel_original<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B, M, C, H, W, K);
	}


	//cudaMemcpyToSymbol(LARGE_MASK, w.dptr_, C * M * K * K * sizeof(float));
	//forward_kernel_original<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B, M, C, H, W, K);
	
	

	// Use MSHADOW_CUDA_CALL to check for CUDAruntime errors.
	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
This tells mxnet how to do an op when it's not a float.
This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
