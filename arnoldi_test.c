#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "arnoldi.h"

/*
 * arnoldi_test.c - David Weir and Teemu Rantalaiho 2013
 *
 *  Copyright 2013 David Weir and Teemu Rantalaiho
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
 *
 *
 * To compile CPU-version:
 *   gcc -O4 arnoldi_test.c carnoldi.c -o arnoldi_test.out -llapack -lm
 * Debug symbols and no optimizations:
 *   gcc -g arnoldi_test.c  carnoldi.c -o arnoldi_test.debug -llapack -lm
 * To compile GPU-version:
 *   nvcc -DCUDA --x cu -O4 -arch=<your arch - for example sm_20> arnoldi_test.c carnoldi.c -o arnoldi_test.gpu -llapack
 * Debug symbols and no optimizations on CPU-side:
 *   nvcc -DCUDA --x cu -g arnoldi_test.c carnoldi.c -o arnoldi_test.dbggpu -llapack
 *
 *   NOTE: with older versions of nvcc it seems that you have to rename the source-files to .cu ending for some odd reason
 *
 * To Compile lib-arcpack++ - version:
 *   g++ -O4 -DARPACKPP arnoldi_test.c carnoldi.c -o arnoldi_test.out_arp -llapack -lm -larpack++ -I<path to lib-arpack++ headers>
 * for example
 *   g++ -O4 -DARPACKPP arnoldi_test.c carnoldi.c -o arnoldi_test.out_arp -llapack -lm -larpack++ -I/usr/include/arpack++
 *
 */



// NOTE: Use parallel algorithms that came with the arnoldi-code
#ifdef MANGLE
#undef MANGLE
#endif
#define MANGLE(X) X
#include "apar_defs.h"

typedef struct mvec_in_s
{
    const scomplex_t* src;
    scomplex_t* dst;
    int size;
} mvec_in;

PARALLEL_KERNEL_BEGIN(matmul_kernel, mvec_in, in, i, skip)
{
    scomplex_t res;
    res.re = in.src[i].re;
    res.im = in.src[i].im;
    if (i > 0){
        res.re += in.src[i-1].re;
        res.im += in.src[i-1].im;
    }
    if (i < in.size - 1)
    {
        res.re += in.src[i+1].re;
        res.im += in.src[i+1].im;
    }
    in.dst[i].re = res.re;
    in.dst[i].im = res.im;
}
PARALLEL_KERNEL_END()

static void matmulfun(void* matdata, const scomplex_t* src, scomplex_t* dst){
    int size = *(int*)matdata;
    mvec_in input;
    input.dst = dst;
    input.src = src;
    input.size = size;
    KERNEL_CALL(matmul_kernel, input, 0, size, 1);
}

#if defined(ARPACKPP)
#include <arscomp.h>
class OurMatrix {
private:
    int size;
public:
    int ncols(void){ return size; }
    void MultMv(arcomplex<float>* src, arcomplex<float>* dst){
        matmulfun((void*)&size, (scomplex_t*)src, (scomplex_t*)dst);
    }
    OurMatrix(int size) { this->size = size; }
};

char* modeToStr(arnmode mode){
    switch(mode){
    case arnmode_LM:
        return "LM";
    case arnmode_LR:
        return "LR";
    case arnmode_SR:
        return "SR";
    default:
        return "SM";
    }
}

int arpackpp_solve(int size, scomplex_t* results, int n_eigs, scomplex_t* initVec, int maxiter, float tol, int n_extend, arnmode mode){
    std::complex<float>* tmpres = (std::complex<float>*)results;
    OurMatrix A(size);
    void (OurMatrix::* mulfunptr) (arcomplex<float>[],arcomplex<float>[]) = &OurMatrix::MultMv;
    ARCompStdEig<float, OurMatrix>myproblem(size, n_eigs, &A, mulfunptr, modeToStr(mode), n_eigs + n_extend, tol, maxiter);
    myproblem.Trace();
    myproblem.Eigenvalues(tmpres, false, false);
}
#endif


static void printUsage(void){
    printf("Simple test to find Eigenvalues of matrix A_ij = delta_ij + delta_i(j+1) + delta_i(j-1)\n\n using the Arnoldi method.");
    printf("Options: \n");
    printf("\t\t --n <Integer>      - Size of the system - default 20 \n");
    printf("\t\t --iter <Integer>   - Max number of Arnoldi iterations - default 100\n");
    printf("\t\t --tol <Float>      - Desired precision - default 1e-5 - NOTE: Solution sought in single-precision.\n");
    printf("\t\t --n_eig <Integer>  - Number of eigenvalues requested - default 4 \n");
    printf("\t\t --n_ext <Integer>  - Number of extended eigenvalues to help the algorithm - default 2 \n");
    printf("\t\t --fast_matmul      - Use a custom (sparse) matrix-vector multiplication function instead of a full matrix\n");
    printf("\t\t --mode <Integer>   - In which mode to run: 1=Largest Magnitude, 2=Largest real part, 3=Small mag, 4=Small real, default=4\n");
    printf("\n\n");
}

int main(int argc, char *argv[]) {

  int i, j;

  int N = 20;

  arnmode mode = arnmode_SR;
  // Custom matrix-vector multiplication disabled by default:
  int fast_matmul = 0;

  // number of eigenvalues
  int n_eigs = 4;

  // final eigenvalues
  scomplex_t* results;

  // number of additional arnoldi iterations
  int n_extend = 2;

  // desired tolerance
  double tol = 1e-5;

  // max number of iterations
  int maxiter = 100;
  int olditer;

  // Possible error
  int error = 0;

  // initialisation vector
  scomplex_t* init_vec;
  scomplex_t* devinit_vec;
  scomplex_t* devmat = NULL;;
  scomplex_t* mat = NULL;

  // create function struct
  arnoldi_abs_int functions;


  printUsage();
  for (i = 1; i < argc; i++){
      if (strcmp(argv[i], "--n") == 0)
          N = atoi(argv[++i]);
      else if (strcmp(argv[i], "--iter") == 0 && argc > i+1)
          maxiter = atoi(argv[++i]);
      else if (strcmp(argv[i], "--tol") == 0 && argc > i+1)
          tol = atof(argv[++i]);
      else if (strcmp(argv[i], "--n_eig") == 0 && argc > i+1)
          n_eigs = atoi(argv[++i]);
      else if (strcmp(argv[i], "--n_ext") == 0 && argc > i+1)
          n_extend = atoi(argv[++i]);
      else if (strcmp(argv[i], "--mode") == 0 && argc > i+1)
          mode = (arnmode)atoi(argv[++i]);
      else if (strcmp(argv[i], "--fast_matmul") == 0)
          fast_matmul = 1;
  }
  olditer = maxiter;

  init_vec = (scomplex_t *)malloc(N*sizeof(scomplex_t));
  results = (scomplex_t *)malloc(n_eigs*sizeof(scomplex_t));

  for(i=0; i<N; i++) {
    // something naive for initial vector
    init_vec[i].re = drand48() - 0.5;
    init_vec[i].im = drand48() - 0.5;
    // Don't worry - this will be normalized by the Arnoldi algorithm
  }

  // tridiagonal matrix - l_s = 1 + 2 cos ( s*pi/ (m+1)) where m is the size of the matrix
  if (!fast_matmul){
      mat = (scomplex_t *)malloc(N*N*sizeof(scomplex_t));
      for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
          if (i == j || (i == j+1) || (i == j-1))
              mat[i*N + j].re = 1.0;
          else
              mat[i*N + j].re = 0.0;
          mat[i*N + j].im = 0.0;
        }
      }
  }

#if defined(CUDA)
  cudaMalloc(&devinit_vec, N*sizeof(scomplex_t));
  if (!fast_matmul) cudaMalloc(&devmat, N*N*sizeof(scomplex_t));
  cudaMemcpy(devinit_vec, init_vec, sizeof(scomplex_t) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(devmat, mat, sizeof(scomplex_t) * N * N, cudaMemcpyHostToDevice);
#else
  devinit_vec = init_vec;
  devmat = mat;
#endif

  // populate function struct
  // use dense matrix for testing
  functions.fullmat.data = fast_matmul ? (void*)&N : devmat;
  functions.fullmat.stride = N;


  // disable reverse communication interface
  functions.mvecmulFun = fast_matmul ? (amatvecmult)matmulfun : NULL;
  functions.allocFieldFun = NULL;
  functions.freeFieldFun = NULL;
  functions.scalar_redFun = NULL;
  functions.complex_redFun = NULL;
  functions.reserve1 = NULL;
  functions.reserve2 = NULL;

  // do it
#if defined(ARPACKPP)
  arpackpp_solve(N, results, n_eigs, init_vec, maxiter, tol, n_extend, mode);
#else
  error =
    run_carnoldi(results, devinit_vec, NULL, N, 1, 0,
                 n_eigs, n_extend, tol, &maxiter, &functions, mode);
#endif
#if !defined(ARPACKPP)
  if (error == 0){
      printf("Arnoldi method complete in %d iterations without errors\n", maxiter);
      if (maxiter == olditer)
          printf("Warning: All iterations exhausted - Arnoldi method didn't converge!\n");
#else
      if (error == 0){
          printf("Arnoldi method complete\n");
#endif
      // print results

      for(i=0; i<n_eigs; i++) {
#if defined(ARPACKPP)
        int j = n_eigs-i-1;
#else
        int j = i;
#endif
        printf("eig %d: %.15f + %.15fi\n", i, results[j].re, results[j].im);
      }
      // print expected results:
      printf("\nExpected results:\n");
      for(i=0; i<n_eigs; i++) {
        int s = N - i;
        float ls;
        if (mode == arnmode_LM || mode == arnmode_LR){
            ls = 1.0 + 2.0 * cos((i+1)*3.141592654/(float)(N+1));
        }
        else if (mode == arnmode_SR) {
            ls = 1.0 + 2.0 * cos(s*3.141592654/(float)(N+1));
        }
        else {
         // Do something clever cos x = -1/2 => x = 2pi/3 => s pi/(n+1) = 2pi/3 => s = 2(n+1)/3
            s = 2*(N+1)/3;
            if (i > 0)
            if ((i & 0x1) == 1)
                s += (i+1)/2;
            else
                s -= (i+1)/2;
            ls = 1.0 + 2.0 * cos(s*3.141592654/(float)(N+1));
        }
        printf("a_eig %d: %.15f + %.15fi\n", i, ls, 0.0);
      }
  } else {
      printf("Arnoldi method ERROR = %d\n", error);
  }

  free(results);
  free(init_vec);
  free(mat);
#if defined(CUDA)
  cudaFree(devmat);
  cudaFree(devinit_vec);
#endif

  return 0;
}
