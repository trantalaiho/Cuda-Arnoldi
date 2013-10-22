/*
 * cuda_matmul.h
 *
 *  Created on: 21.5.2013
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2013 Teemu Rantalaiho
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *
 *
 */
#ifndef CUDA_MATMUL_H_
#define CUDA_MATMUL_H_

#include <cuda_runtime_api.h>

template <typename OUTPUTTYPE, typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename SRCTYPE, typename DSTTYPE>
static inline
cudaError_t callFullMatMul(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
    INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
    bool transpose = false, cudaStream_t stream = 0, bool outInDev = true);


template <typename INDEXTYPE, typename RADIXTYPE>
static inline
cudaError_t callFloatMatMul(
    const RADIXTYPE* mat, INDEXTYPE sizex, INDEXTYPE sizey, INDEXTYPE stride,
    const RADIXTYPE* src, RADIXTYPE* result, bool transpose = false,
    cudaStream_t stream = 0, bool outInDev = true);




#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __device__
#define __device__
#endif



#define MM_BLOCKSIZE_LOG2   7
#define MM_BLOCKSIZE        (1<<MM_BLOCKSIZE_LOG2)

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename OUTPUTTYPE, typename SRCTYPE, typename DSTTYPE>
__global__
void callFullMatmulKernel(
        INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
        INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
        int nstepsX, INDEXTYPE starty = 0)
{
    int tid = threadIdx.x;
    INDEXTYPE y = (INDEXTYPE)(blockIdx.x + starty);
    INDEXTYPE x = (INDEXTYPE)tid;
    OUTPUTTYPE myRes;
    if (x < sizex && y < sizey){
        myRes = xformFunctor(input, x, y, src);
        x += MM_BLOCKSIZE;
    }

#define NUNROLL_LOG2 2
#define NUNROLL (1 << NUNROLL_LOG2)
    int nFullSteps = (nstepsX - 1) >> NUNROLL_LOG2;
    for (int fstep = 0; fstep < nFullSteps; fstep++){
#pragma unroll
        for (int substep = 0; substep < NUNROLL; substep++){
            OUTPUTTYPE tmpres = xformFunctor(input, x, y, src);
            myRes = sumFunctor(myRes, tmpres);
            x += MM_BLOCKSIZE;
        }
    }
    while (x < sizex){
        OUTPUTTYPE tmpres = xformFunctor(input, x, y, src);
        myRes = sumFunctor(myRes, tmpres);
        x += MM_BLOCKSIZE;
    }
    {
        __shared__ OUTPUTTYPE tmparr[MM_BLOCKSIZE];
        tmparr[tid] = myRes;
        __syncthreads();
        if (sizex < MM_BLOCKSIZE){
            if (tid == 0){
                for (int i = 1; i < (sizex < MM_BLOCKSIZE ? sizex : MM_BLOCKSIZE); i++)
                    myRes = sumFunctor(myRes, tmparr[i]);
            }
        } else {
            if (tid < 32){
#pragma unroll
                for (int i = 1; i < (MM_BLOCKSIZE >> 5); i++)
                    myRes = sumFunctor(myRes, tmparr[tid + (i<<5)]);
                tmparr[tid] = myRes;
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+16]);
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+8]);
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+4]);
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+2]);
                __threadfence_block();
                myRes = sumFunctor(tmparr[tid], tmparr[tid+1]);
            }
        }
    }
    if (threadIdx.x == 0){
        storeDstFun(result, y, myRes);
    }
}

static inline int divLog2RoundUp(int size, int divlog2)
{
    int div = 1 << divlog2;
    int paddedSize = (size + div - 1) & (~(div - 1));
    int res = paddedSize >> divlog2;
    return res;
}

#include <stdio.h>

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename OUTPUTTYPE, typename SRCTYPE, typename DSTTYPE>
static
cudaError_t callFullMatMulImpl(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
    INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
    cudaStream_t stream, bool outInDev)
{
  //int stepsX = divLog2RoundUp(sizex, MM_BLOCKSIZE_LOG2);
    int stepsX = sizex >> MM_BLOCKSIZE_LOG2;
    dim3 grid = sizey;
    dim3 block = MM_BLOCKSIZE;
    if (!outInDev){
        printf("Sorry - no support yet for CPU-output buffers...\n");
        return cudaSuccess;
    }
    if (sizex <= 0 || sizey <= 0)
        return cudaSuccess;
/*    printf("block = (%d, %d, %d), grid = (%d,%d)\n", block.x, block.y, block.z, grid.x, grid.y);*/
    if (sizey > 32768){
        grid = 32768;
        int startY = 0;
        while (startY < sizey){
            if (startY + 32768 > sizey)
                grid = sizey - startY;
            callFullMatmulKernel
                    <TRANSFORMFUNTYPE, INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
                    <<<grid, block,0,stream>>>
                    (input, xformFunctor, sumFunctor, storeDstFun,
                    sizex, sizey, src, result, stepsX, startY);
            startY += 32768;
        }
    }
    else
    {
        callFullMatmulKernel
                <TRANSFORMFUNTYPE, INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
                <<<grid, block,0,stream>>>
                (input, xformFunctor, sumFunctor, storeDstFun, sizex, sizey, src, result, stepsX);
    }
    return cudaGetLastError();
}

template <typename INDEXTYPE, typename OUTPUTTYPE, typename TRANSFORMFUNTYPE, typename INPUTTYPE, typename SRCTYPE>
struct FullMatMulTransposeWrapper {
	TRANSFORMFUNTYPE userFunctor;
	inline __device__
    OUTPUTTYPE operator()( INPUTTYPE input, INDEXTYPE x, INDEXTYPE y, SRCTYPE src){
        return userFunctor(input, y, x, src);
    }
};


template <typename OUTPUTTYPE, typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename SRCTYPE, typename DSTTYPE>
cudaError_t callFullMatMul(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
    INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
    bool transpose, cudaStream_t stream, bool outInDev)
{
	cudaError_t err;
	if (transpose){
		struct FullMatMulTransposeWrapper<INDEXTYPE, OUTPUTTYPE, TRANSFORMFUNTYPE,INPUTTYPE,SRCTYPE> wrapperFun;
		wrapperFun.userFunctor = xformFunctor;
		err = callFullMatMulImpl
		        <FullMatMulTransposeWrapper<INDEXTYPE, OUTPUTTYPE, TRANSFORMFUNTYPE,INPUTTYPE,SRCTYPE>,
		            INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
		        (input, wrapperFun, sumFunctor, storeDstFun, sizey, sizex, src, result, stream, outInDev);
	}
	else
	{

	    err = callFullMatMulImpl
	            <TRANSFORMFUNTYPE, INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
		        (input, xformFunctor, sumFunctor, storeDstFun, sizex, sizey, src, result, stream, outInDev);
	}
	return err;
}


template <typename INDEXTYPE, typename RADIXTYPE>
struct floatMatMulFun {
    INDEXTYPE stride;
    __device__
    RADIXTYPE operator()( const RADIXTYPE* data, INDEXTYPE x, INDEXTYPE y, const RADIXTYPE* src){
        RADIXTYPE m_ij = data[y*stride + x];
        RADIXTYPE res = m_ij*src[x];
        return res;
    }
};

template <typename RADIXTYPE>
struct radixSumFun {
    __device__
    RADIXTYPE operator()( RADIXTYPE a, RADIXTYPE b){
        RADIXTYPE res = a + b;
        return res;
    }
};
template <typename INDEXTYPE, typename RADIXTYPE>
struct storefloatFun {
    __device__
    void operator()( RADIXTYPE* result, INDEXTYPE i, RADIXTYPE var){
        result[i] = var;
    }
};



template <typename INDEXTYPE, typename RADIXTYPE>
cudaError_t callFloatMatMul(
    const RADIXTYPE* mat, INDEXTYPE sizex, INDEXTYPE sizey, INDEXTYPE stride,
    const RADIXTYPE* src, RADIXTYPE* result, bool transpose,
    cudaStream_t stream, bool outInDev)
{
    struct floatMatMulFun<INDEXTYPE, RADIXTYPE> mulfun;
    struct radixSumFun<RADIXTYPE> sumfun;
    struct storefloatFun<INDEXTYPE, RADIXTYPE> storeDstfun;
    mulfun.stride = stride;
    cudaError_t err = callFullMatMul<RADIXTYPE, floatMatMulFun<INDEXTYPE, RADIXTYPE>, INDEXTYPE, const RADIXTYPE*, radixSumFun<RADIXTYPE>,
                                     storefloatFun<INDEXTYPE, RADIXTYPE>, const RADIXTYPE*, RADIXTYPE*>
                        (mat, mulfun, sumfun, storeDstfun, sizex, sizey, src, result, transpose, stream, outInDev);
    if (err != cudaSuccess)
    	printf("Error in callFullMatMul! err = %s\n", cudaGetErrorString(err));
    return err;
}



#endif /* CUDA_MATMUL_H_ */
