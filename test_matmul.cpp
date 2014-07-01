/*
 * test_matmul.cpp
 *
 *  Created on: 29.5.2013
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


// Compile: nvcc -O4 -x=cu -arch=<your arch> test_matmul.cpp -o test_matmul -lcudart
// Example: nvcc -O4 -x=cu -arch=sm_12 test_matmul.cpp -o test_matmul -lcudart

// To compare against cublas, compile with:
// Compile: nvcc -O4 -x=cu -arch=<your arch> test_matmul.cpp -o test_matmul -DCUBLAS -lcublas -lcudart
// Example: nvcc -O4 -x=cu -arch=sm_12 test_matmul.cpp -o test_matmul -DCUBLAS -lcublas -lcudart



#include "cuda_matmul.h"
#include <assert.h>
#include <stdio.h>

#include <cuda_runtime.h>

#ifdef CUBLAS
#  define USE_CUBLAS    1
#else
#  define USE_CUBLAS    0
#endif

#if USE_CUBLAS
#include <cublas_v2.h>
cublasHandle_t s_cbHandle;
cublasStatus_t s_cbStat;
#endif

#define TESTMAXVAL      (1<<20)

#define TESTMAX_X       (1024)
#define TESTMAX_Y       (1024)

#define DEFSIZEX        933
#define DEFSIZEY        851

#define NRUNS           100

#ifdef DOUBLE
#define radix   double
#else
#define radix   float
#endif

static inline radix getIntInput(int i, bool rnd){
    static unsigned int current = 0xf1232345;
    const unsigned int mult = 1664525;
    const unsigned int add = 1013904223ul;

    current = current * mult + add;
    if (!rnd)
      current = i / 100;
    i = (int)(current & (TESTMAXVAL -1));
    return i;

}


template <typename T>
static inline T getInput(int i, bool rnd);

template <>
static inline radix getInput(int i, bool rnd)
{
    return ((radix)getIntInput(i,rnd))/((radix)TESTMAXVAL);
}
template <>
static inline int getInput(int i, bool rnd)
{
    return getIntInput(i,rnd);
}



template <typename T>
static void fillInput(T* input, bool rnd, int totsize)
{
  int i;
  for (i = 0; i < totsize;)
    *input++ = getInput<T>(i++, rnd);
}



#include <sys/time.h>

static double cputime_fast()
{
  struct timeval resource;
  gettimeofday(&resource,NULL);
  return( resource.tv_sec + 1.0e-6*resource.tv_usec );
}

void printUsage(void)
{
  printf("\n");
  printf("Test matmul\n\n");
  printf("\tOptions:\n\n");
  printf("\t\t--cpu\t\t Run on CPU serially instead of GPU\n");
  printf("\t\t--print\t\t Print results of algorithm (check validity)\n");

  printf("\t\t--rnd\t Take uniform random keys\n");
  printf("\t\t--int\t Use integer matrix\n");
  printf("\t\t--stress <nruns> <seed>\t Run stress-tests with given number of runs and seed (max size with --size).\n");
  printf("\t\t--size x y\t Use size x y problem, otherwise it's (%d, %d)\n", DEFSIZEX, DEFSIZEY);
  printf("\t\t--cublas\t Use CUBLAS instead of GPU to multiply - note: Compile with -DCUBLAS -lcublas\n");
}

template<typename T>
void printVec(const char* name, T* vec, int size);


template<>
void printVec(const char* name, radix* vec, int size){
    printf("%s = \n[ ", name);
    for (int i = 0; i < size ; i++)
        printf(" %e%s ", vec[i], i < size-1 ? "," : " ]\n\n");
}

template<>
void printVec(const char* name, int* vec, int size){
    printf("%s = \n[ ", name);
    for (int i = 0; i < size ; i++)
        printf(" %d%s ", vec[i], i < size-1 ? "," : " ]\n\n");
}

template<typename SCALART>
bool checkRes(SCALART a, SCALART b, int sizex, int sizey, int index);

template<>
bool checkRes(radix a, radix b, int sizex, int sizey, int index)
{
    if (fabs(a-b)/(a+b+0.1)*((double)sizex) > 1.e-9 ){
        printf("CPU and GPU results differ! CPU = %e, GPU = %e, diff = %e, index = %d , size = (%d, %d)\n",
                a, b, a - b, index, sizex, sizey);
        return false;
    }
    return true;
}
template<>
bool checkRes(int a, int b, int sizex, int sizey, int index)
{
    if (a != b){
        printf("CPU and GPU results differ! CPU = %d, GPU = %d, diff = %d, index = %d , size = (%d, %d)\n",
                a, b, a - b, index, sizex, sizey);
        return false;
    }
    return true;
}


#if USE_CUBLAS
template <typename T>
static inline void cublasMatmul(T* matrix, int sizex, int sizey, int stride, T* src, T* dst);

template <>
static inline void cublasMatmul(float* matrix, int sizex, int sizey, int stride, float* src, float* dst){
    const float alpha = 1.0;
    const float beta = 0.0;
    s_cbStat =
    cublasSgemv(s_cbHandle, CUBLAS_OP_T, sizex, sizey,
            &alpha,                     // alpha
            matrix, stride,             // matrix, stride
            src, 1,                     // src vector, stride
            &beta,                      // beta
            dst, 1                      // dst vector y and stride:  y = alpha Ax + Beta y
            );
}
template <>
static inline void cublasMatmul(int* matrix, int sizex, int sizey, int stride, int* src, int* dst){
    printf("Sorry - CUBLAS can't handle integer matrices!!! (No blas can...)\n");
}
template <>
static inline void cublasMatmul(double* matrix, int sizex, int sizey, int stride, double* src, double* dst){
    const double alpha = 1.0;
    const double beta = 0.0;
    s_cbStat =
    cublasDgemv(s_cbHandle, CUBLAS_OP_T, sizex, sizey,
            &alpha,                         // alpha
            matrix, stride,                 // matrix, stride
            src, 1,                         // src vector, stride
            &beta,                          // beta
            dst, 1                          // dst vector y and stride:  y = alpha Ax + Beta y
            );

}
#endif


template <typename SCALART>
bool testMatMul (
        SCALART* INPUT, SCALART* inputVec, SCALART* OUTPUT,
        SCALART* hostINPUT, SCALART* hostVec, SCALART* hostOUTPUT,
        int sizex, int sizey, bool print, bool cpu, bool check, bool cublas, bool transpose)
{
    if (cpu || check)
    {
        if (transpose){
            if (sizey > 0) for (int i=0;i<sizex;i++){
                hostOUTPUT[i] = 0;
                for (int k = 0; k < sizey; k++)
                    hostOUTPUT[i] += hostINPUT[i + k * sizex] * hostVec[k];
            }
        }
        else
        {
            if (sizex > 0) for (int i = 0; i < sizey; i++){
                hostOUTPUT[i] = 0;
                for (int k = 0; k < sizex; k++)
                    hostOUTPUT[i] += hostINPUT[i * sizex + k] * hostVec[k];
            }
        }
    }

    SCALART* tmp;
    if (!cpu){
        cudaError_t error = cudaSuccess;
        //cudaMemset(OUTPUT, 0x00, sizeof(SCALART) * sizey);
        #if USE_CUBLAS
        if (cublas){
            // cublasSetStream(stream);
            cublasMatmul(INPUT, sizex,sizey,sizex,inputVec,OUTPUT);
            if (s_cbStat != CUBLAS_STATUS_SUCCESS){
                printf ("CUBLAS sgemv failed\n");
            }
        }
        else
        #endif
        {
            error = callFloatMatMul(INPUT, sizex, sizey, sizex, inputVec, OUTPUT, transpose, 0, true);
        }
        if (error != cudaSuccess)
            printf("CUDA ERROR = %s\n", cudaGetErrorString(error));
        if (check){
            tmp = (SCALART*)malloc(sizeof(SCALART) * sizey);
            cudaMemcpy(tmp, OUTPUT, sizeof(SCALART) * sizey, cudaMemcpyDeviceToHost);
        }
    }

    if (print){
        if (cpu){
            printVec("result[cpu]", hostOUTPUT, sizey);
        } else {
            printVec("result[gpu]", tmp, sizey);
        }
    }

    if (check && !cpu){
        for (int i = 0; i < (transpose ? sizex : sizey); i++){
            if (!checkRes(hostOUTPUT[i], tmp[i], sizex, sizey, i))
                return false;
        }
        free(tmp);
    }
    return true;
}


template <typename SCALART>
struct dataT
{
    SCALART* input;
    SCALART* vec;
    SCALART* output;
    SCALART* h_input;
    SCALART* h_vec;
    SCALART* h_output;
};

template <typename SCALART>
void initInput(bool gpu, struct dataT<SCALART> *result, int sizex, int sizey, bool rnd)
{
    // Allocate numbers:
    SCALART* INPUT = NULL;
    SCALART* OUTPUT = NULL;
    int totsize = (sizex * (sizey+1));
    SCALART* hostINPUT = (SCALART*)malloc(sizeof(SCALART) * totsize );
    SCALART* hostOUTPUT = (SCALART*)malloc(sizeof(SCALART) * sizey );
    assert(hostINPUT);
    assert(hostOUTPUT);

    fillInput(hostINPUT, rnd, totsize);
    if (gpu)
    {
      cudaMalloc(&INPUT, sizeof(SCALART) * totsize);
      cudaMalloc(&OUTPUT, sizeof(SCALART) * sizey );
      assert(INPUT);
      cudaMemcpy(INPUT, hostINPUT, sizeof(SCALART) * totsize, cudaMemcpyHostToDevice);
    }
    SCALART* hostVec = &hostINPUT[sizex * sizey];
    SCALART* inputVec = &INPUT[sizex * sizey];
    result->input = INPUT;
    result->vec = inputVec;
    result->output = OUTPUT;
    result->h_input = hostINPUT;
    result->h_vec = hostVec;
    result->h_output = hostOUTPUT;
}

static unsigned int s_current = 0;
void seedRnd(unsigned int seed){
    s_current = seed;
}

unsigned int getRnd(void){
    const unsigned int a = 0xDDB3D742u;
    const unsigned int b = 0xB17217F7u;
    unsigned int tmp1, tmp2;
    tmp1 = (s_current + b) * a;
    tmp2 = (tmp1 + b) * a;
    s_current = ((tmp1 & 0xffffff00u) << 8) | ((tmp2 & 0x00ffffffu) >> 8);
    return s_current;
}

void simpleTest(void){
    printf("First running a very simple test:\n\n\n");

    printf("    ( 1 2 )\n");
    printf("A = ( 3 4 )\n");
    printf("    ( 5 6 )\n\n");

    printf("    ( -1  2  1  5 )\n");
    printf("B = (  3 -4  2  3 )\n");
    printf("    (  5 -6  1  2 )\n\n");


    printf("x1 = ( 2 )\n");
    printf("     ( 3 )\n\n");
    printf("      ( 8  )\n");
    printf("Ax1 = ( 18 )\n");
    printf("      ( 28 )\n\n");

    printf("     ( 3 )\n");
    printf("x2 = ( 5 )\n");
    printf("     ( 7 )\n\n");
    printf("A^T x2 = ( 53 )\n");
    printf("         ( 68 )\n\n");


    printf("     ( 2 )\n");
    printf("x3 = ( 0 )\n");
    printf("     ( 0 )\n\n");
    printf("A^T x3 = ( 2 )\n");
    printf("         ( 4 )\n\n");

    printf("     ( 3 )\n");
    printf("x4 = ( 5 )\n");
    printf("     ( 2 )\n");
    printf("     ( 1 )\n\n");

    printf("      ( 14 )\n");
    printf("Bx4 = ( -4 )\n");
    printf("      (-11 )\n\n");

    printf("        ( 47 )\n");
    printf("B^Tx2 = (-56 )\n");
    printf("        ( 20 )\n");
    printf("        ( 44 )\n\n");

    printf("Ok - now trying:\n\n");

    {
        radix *m, *x1, *x2, *x3, *res, *m2, *x4;
        radix A[] = { 1.0, 2.0,   3.0, 4.0,   5.0, 6.0 };
        // NOTE: stride = 5

        radix B[] = { -1.0, 2.0,  1.0, 5.0,   -111.0, 10000,
                       3.0, -4.0, 2.0, 3.0,  -2222.0, 51133,
                       5.0, -6.0, 1.0, 2.0,   -3.2, 0.2 };
        /*radix B[] = { -1.0, 2.0,  1.0, 5.0,
                       3.0, -4.0, 2.0, 3.0,
                       5.0, -6.0, 1.0, 2.0};*/
        radix h1[] = { 2.0, 3.0};
        radix h2[] = { 3.0, 5.0, 7.0 };
        radix h3[] = { 2.0, 0.0, 0.0 };
        radix h4[] = { 3.0, 5.0, 2.0, 1.0 };
        radix hres[] = { 3.0, 5.0, 7.0, 2.0 };
        cudaMalloc(&m, sizeof(A));
        cudaMalloc(&m2, sizeof(B));
        cudaMalloc(&x1, sizeof(h1));
        cudaMalloc(&x2, sizeof(h2));
        cudaMalloc(&x3, sizeof(h3));
        cudaMalloc(&x4, sizeof(h4));
        cudaMalloc(&res, sizeof(hres));

        cudaMemcpy(m, A, sizeof(A), cudaMemcpyHostToDevice);
        cudaMemcpy(m2, B, sizeof(B), cudaMemcpyHostToDevice);

        cudaMemcpy(x1, h1, sizeof(h1), cudaMemcpyHostToDevice);
        cudaMemcpy(x2, h2, sizeof(h2), cudaMemcpyHostToDevice);
        cudaMemcpy(x3, h3, sizeof(h3), cudaMemcpyHostToDevice);
        cudaMemcpy(x4, h4, sizeof(h4), cudaMemcpyHostToDevice);
        // x1
        cudaMemset(res, -1, sizeof(hres));
        callFloatMatMul(m, 2, 3, 2, x1, res, false);
        cudaMemcpy(hres, res, sizeof(hres), cudaMemcpyDeviceToHost);
        printf("Ax1 = [ %f, %f, %f ]\n", hres[0], hres[1], hres[2]);
        // x2
        cudaMemset(res, -1, sizeof(hres));
        callFloatMatMul(m, 2, 3, 2, x2, res, true);
        cudaMemcpy(hres, res, sizeof(hres), cudaMemcpyDeviceToHost);
        printf("Ax2 = [ %f, %f]\n", hres[0], hres[1]);
        // x3
        cudaMemset(res, -1, sizeof(hres));
        callFloatMatMul(m, 2, 3, 2, x3, res, true);
        cudaMemcpy(hres, res, sizeof(hres), cudaMemcpyDeviceToHost);
        printf("Ax3 = [ %f, %f]\n\n\n", hres[0], hres[1]);

        // Bx4
        cudaMemset(res, -1, sizeof(hres));
        callFloatMatMul(m2, 4, 3, 6, x4, res, false);
        cudaMemcpy(hres, res, sizeof(hres), cudaMemcpyDeviceToHost);
        printf("Bx4 = [ %f, %f, %f ]\n", hres[0], hres[1], hres[2]);

        // B^Tx2
        cudaMemset(res, -1, sizeof(hres));
        callFloatMatMul(m2, 4, 3, 6, x2, res, true);
        cudaMemcpy(hres, res, sizeof(hres), cudaMemcpyDeviceToHost);
        printf("B^T x2 = [ %f, %f, %f, %f ]\n", hres[0], hres[1], hres[2], hres[3]);


        cudaFree(m);
        cudaFree(m2);
        cudaFree(x1);
        cudaFree(x2);
        cudaFree(x3);
        cudaFree(x4);
        cudaFree(res);
    }
}




int main (int argc, char** argv)
{
  int i;

  bool cpu = false;
  bool print = false;

  bool rnd = false;
  bool useint = false;
  double t0;

  bool stress = false;
  bool cublas = false;
  bool transpose = false;
  bool simple = false;


  int sizex = DEFSIZEX;
  int sizey = DEFSIZEY;
  int seed = 0;
  int nruns = NRUNS;

  printUsage();

  for (i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--cpu") == 0)
      cpu = true;
    if (argv[i] && strcmp(argv[i], "--print") == 0)
      print = true;
    if (argv[i] && strcmp(argv[i], "--rnd") == 0)
        rnd = true;
    if (argv[i] && strcmp(argv[i], "--int") == 0)
        useint = true;
    if (argv[i] && strcmp(argv[i], "--cublas") == 0)
        cublas = true;
    if (argv[i] && strcmp(argv[i], "--transpose") == 0)
        transpose = true;
    if (argv[i] && strcmp(argv[i], "--simple") == 0)
        simple = true;
    if (argv[i] && strcmp(argv[i], "--stress") == 0){
        if (i + 2 < argc){
            stress = true;
            nruns = atoi(argv[++i]);
            seed = atoi(argv[++i]);
            seedRnd(seed);
            if (nruns <= 0)
                return -4;
        }
        else {
            return -3;
        }
    }
    if (argv[i] && strcmp(argv[i], "--size") == 0){
        if (i + 2 < argc){
            sizex = atoi(argv[++i]);
            sizey = atoi(argv[++i]);
            if (sizex <= 0 || sizey <= 0)
                return -2;
        }
        else
        {
            return -1;
        }
    }
  }

  if (simple)
      simpleTest();


  {
    // Allocate numbers:
    struct dataT<int> resInt;
    struct dataT<radix> resRadix;
    if (useint)
        initInput(!cpu, &resInt, sizex, sizey, rnd);
    else
        initInput(!cpu, &resRadix, sizex, sizey, rnd);
    bool check = true;
#if USE_CUBLAS
    if (cublas){
        s_cbStat = cublasCreate(&s_cbHandle);
            if (s_cbStat != CUBLAS_STATUS_SUCCESS) {
                printf ("CUBLAS initialization failed\n");
        return -6;
        }

    }
#else
    if (cublas){
        printf("Requesting cublas without support for cublas compiled! compile with -DCUBLAS -lcublas !!!\n");
        return -7;
    }
#endif
    // Now start timer:
    t0 = cputime_fast();

    if (stress){
        for (i = 0; i < nruns; i++){
            bool success;
            bool transp = transpose ? (getRnd() != 0) : false;
            int localSizex = getRnd() % (!transp ? sizex : sizey);
            int localSizey = getRnd() % (!transp ? sizey : sizex);
            if (useint)
                success = testMatMul(resInt.input, resInt.vec, resInt.output, resInt.h_input, resInt.h_vec, resInt.h_output, localSizex, localSizey, false, cpu, check, cublas, transpose);
            else
                success = testMatMul(resRadix.input, resRadix.vec, resRadix.output, resRadix.h_input, resRadix.h_vec, resRadix.h_output, localSizex, localSizey, false, cpu, check, cublas, transpose);
            if (!cpu)
                cudaStreamSynchronize(0);
            printf(".");
            fflush(stdout);
            if (!success)
                break;
        }
    }
    else {
        for (i = 0; i < NRUNS; i++)
        {
          bool success;
          if (useint)
              success = testMatMul(resInt.input, resInt.vec, resInt.output, resInt.h_input, resInt.h_vec, resInt.h_output, !transpose ? sizex : sizey, !transpose ? sizey : sizex, print, cpu, check, cublas, transpose);
          else
              success = testMatMul(resRadix.input, resRadix.vec, resRadix.output, resRadix.h_input, resRadix.h_vec, resRadix.h_output, !transpose ? sizex : sizey, !transpose ? sizey : sizex, print, cpu, check, cublas, transpose);
          if (!cpu)
              cudaStreamSynchronize(0);
          if (!success && check) printf("ERROR IN COMPUTATION - treat perf results with skepticism!!..\n\n");
          print = false;
          check = false;
          // Run only once all stress-tests
        }
    }
    {
        double t = cputime_fast() - t0;
        printf("Runtime in loops: %fs\n", t);

        if (!stress){
            double GFlops = ((double)sizex*sizey*2*NRUNS)*1.e-9;
            double GFlops_ps = GFlops / (double)t;
            double GB_ps = GFlops_ps * (double)(useint ? 4 : sizeof(radix));
            printf("# Throughput (GFLOPS/s), Throughput (GB/s)\n");
            printf("%4f,\t\t%4f\n", GFlops_ps, GB_ps);
        }
    }
    if (useint){
        if (resInt.input) cudaFree(resInt.input);
        if (resInt.output) cudaFree(resInt.output);
        if (resInt.h_input) free(resInt.h_input);
        if (resInt.h_output) free(resInt.h_output);
    }
    else
    {
        if (resRadix.input) cudaFree(resRadix.input);
        if (resRadix.output) cudaFree(resRadix.output);
        if (resRadix.h_input) free(resRadix.h_input);
        if (resRadix.h_output) free(resRadix.h_output);
    }
  }
  return 0;
}

