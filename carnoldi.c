/*
 * carnoldi.c
 *
 *  Created on: 7.8.2013
 *      Author: Teemu Rantalaiho
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
 */

#include "arnoldi.h"

#ifdef TIMER
#define TRACE_STATISTICS	1
#include <sys/time.h>
#include <sys/resource.h>
double cputime(void)
{
  struct rusage resource;
  //extern int getrusage();
  extern int getrusage(int who, struct rusage *usage);

  getrusage(RUSAGE_SELF,&resource);
  return(resource.ru_utime.tv_sec + 1e-6*resource.ru_utime.tv_usec +
         resource.ru_stime.tv_sec + 1e-6*resource.ru_stime.tv_usec);

}

#endif


#define RADIX_SIZE 32

// For normal complex types
typedef struct fieldtype_s
{
   float re;
   float im;
} fieldtype;
typedef struct fieldentry_s
{
    float re;
    float im;
} fieldentry;

#define MANGLE(X) carnoldi_##X
#include "arnoldi_generic.h"

#ifndef PRINT_FREE
#define PRINT_FREE(X)
#endif

#ifndef PRINT_MALLOC
#define PRINT_MALLOC(X)
#endif

// static cputimefunT s_cputimef = (cputimefunT)0;

static arnoldi_abs_int s_functions;


fieldtype* new_fieldtype(int size, int nMulti){
    fieldtype* result;
    if (s_functions.allocFieldFun){
        result = (fieldtype*)s_functions.allocFieldFun(size, nMulti);
    }
    else {
#ifdef CUDA
        cudaMalloc(&result, nMulti * size * sizeof(fieldentry));
        PRINT_MALLOC(result);
#else
        result = (fieldtype*)s_mallocf(nMulti * size * sizeof(fieldentry));
#endif
    }
    return result;
}
void free_fieldtype(fieldtype* f){
    if (s_functions.freeFieldFun){
        s_functions.freeFieldFun(f);
    }
    else {
#ifdef CUDA
        cudaFree(f);
        PRINT_FREE(f);
#else
        s_freef(f);
#endif
    }
}

__device__
void get_fieldEntry(const fieldtype* f, int i, fieldentry* result, int multiIdx, int stride){
    fieldtype tmp = f[i+multiIdx*stride];
    result->re = tmp.re;
    result->im = tmp.im;
}
__device__
void set_fieldEntry(fieldtype* f, int i, const fieldentry* entry, int multiIdx, int stride){
    fieldtype tmp;
    tmp.re = entry->re;
    tmp.im = entry->im;
    f[i+multiIdx*stride] = tmp;
}
__device__
lcomplex fieldEntry_dot(const fieldentry* a, const fieldentry* b){
    lcomplex res;
    // (z1*) * z2 = (x1 - iy1)*(x2 + iy2) = x1x2 + y1y2 + i(x1y2 - y1x2)
    res.real = a->re * b->re + a->im * b->im;
    res.imag = a->re * b->im - a->im * b->re;
    return res;
}
__device__
radix fieldEntry_rdot(const fieldentry* a, const fieldentry* b){
    return a->re * b->re + a->im * b->im;
}
// dst = scalar * a
__device__
void fieldEntry_scalar_mult(const fieldentry* a, radix scalar, fieldentry* dst ){
    dst->re = scalar * a->re;
    dst->im = scalar * a->im;
}
// dst = a + scalar * b
__device__
void fieldEntry_scalar_madd(const fieldentry * restrict a, radix scalar, const fieldentry * restrict b, fieldentry * restrict dst ){
    dst->re = a->re + scalar * b->re;
    dst->im = a->im + scalar * b->im;
}
// dst = scalar * a
__device__
void fieldEntry_complex_mult(const fieldentry* a, lcomplex scalar, fieldentry* dst ){
    dst->re = scalar.real * a->re - scalar.imag * a->im;
    dst->im = scalar.real * a->im + scalar.imag * a->re;
}
// dst = a + scalar * b
__device__
void fieldEntry_complex_madd(const fieldentry * restrict a, lcomplex scalar, const fieldentry * restrict b, fieldentry * restrict dst ){
    dst->re = a->re + scalar.real * b->re - scalar.imag * b->im;
    dst->im = a->im + scalar.real * b->im + scalar.imag * b->re;
}
typedef struct init_arn_vec_in_s
{
    const fieldtype* src;
    fieldtype* dst;
    int stride;
} init_arn_vec_in;

PARALLEL_KERNEL_BEGIN(init_arn_vec, init_arn_vec_in, in, i, multiIdx)
{
    fieldentry x;
    if (in.src){
        get_fieldEntry(in.src, i, &x, multiIdx, in.stride);
    }
    else {
        if (i == 0)
            x.re = 1.0;
        else
            x.re = 0.0;
        x.im = 0.0;
    }
    set_fieldEntry(in.dst, i, &x, multiIdx, in.stride);
}
PARALLEL_KERNEL_END()

static int s_size = 0;

#ifdef CUDA
#include "cuda_matmul.h"

template <typename INDEXTYPE>
struct complexMatMulFun {
    INDEXTYPE stride;
    __device__
    scomplex_t operator()( const scomplex_t* data, INDEXTYPE x, INDEXTYPE y, scomplex_t* src){
        scomplex_t m_ij = data[y*stride + x];
        scomplex_t res;
        scomplex_t a = src[x];
        res.re = m_ij.re * a.re - m_ij.im * a.im;
        res.im = m_ij.re * a.im + m_ij.im * a.re;
        return res;
    }
};

struct complexSumFun {
    __device__
    scomplex_t operator()( scomplex_t a, scomplex_t b){
        scomplex_t res;
        res.re = a.re + b.re;
        res.im = a.im + b.im;
        return res;
    }
};
template <typename INDEXTYPE, typename ARRAYENTRYT>
struct complexStoreFun {
    __device__
    void operator()( ARRAYENTRYT* dst, INDEXTYPE i, ARRAYENTRYT entry){
        dst[i] = entry;
    }
};

#endif

static
int backup_matmul(void* matctx, void* src, void* dst){
    int error       = 0;
    int size        = s_size;
    fullmat_t* mat  = (fullmat_t*)matctx;
    scomplex_t* m   = (scomplex_t*)mat->data;
    scomplex_t* x0  = (scomplex_t*)src;
    scomplex_t* y   = (scomplex_t*)dst;
#ifndef CUDA
    // Trivial implementation - just for backup, replace at will with sgemv
    int i;
    int hopstride = mat->stride - size;
    scomplex_t* lim = x0 + size;
    for (i = 0; i < size; i++){
        scomplex_t* x = x0;
        scomplex_t res;
        res.im = 0.0;
        res.re = 0.0;
        while (x < lim){
            scomplex_t a = *m++;
            scomplex_t b = *x++;
            res.re += a.re * b.re - a.im * b.im;
            res.im += a.re * b.im + a.im * b.re;
        }
        *y++ = res;
        m += hopstride;
    }
#else
    complexMatMulFun<int> matmulf;
    complexStoreFun<int, scomplex_t> storef;
    complexSumFun cmplxsumf;
    matmulf.stride = mat->stride;
    error = (int)callFullMatMul<scomplex_t>(m, matmulf, cmplxsumf, storef, size, size, x0, y, false, 0, true);
#endif
    return error;
}

// Note - with a normal vector of complex numbers, use nMulti = 1, stride = 0
int run_carnoldi(
        scomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride,
        int n_eigs, int n_extend, double tolerance, int* maxIter,
        const arnoldi_abs_int* functions, arnmode mode)
{
    int error = 0;

    fieldtype** e_vecs = (fieldtype**)s_mallocf(sizeof(fieldtype*) * (n_eigs + n_extend));
    scomplex_t* e_vals = (scomplex_t*)s_mallocf(sizeof(scomplex_t) * (n_eigs + n_extend));
    int i;
    if (!(e_vecs && e_vals)){
        error = -1;
        goto cleanup;
    }
    s_size = size;
    s_functions = *functions;
    for (i = 0; i < n_eigs + n_extend; i++){
        // TODO: Handle allocation errors
        e_vecs[i] = new_fieldtype(size, nMulti);
        if (!e_vecs[i])
            error = -2;
    }
    if (error)
        goto cleanup;
    {
        init_arn_vec_in in;
        in.dst = e_vecs[0];
        in.src = (const fieldtype*)init_vec;
        in.stride = stride;
        KERNEL_CALL(init_arn_vec, in, 0, size, nMulti);
    }
    // Set necessary function pointers

    s_mulf = (mv_mul_t)functions->mvecmulFun;
    if (!s_mulf){
        if (!functions->fullmat.data){
            error = -1001;
            goto cleanup;
        }
        s_mulf = (mv_mul_t)backup_matmul;
        s_matmul_cxt = (void*)&functions->fullmat;
    }
    else{
        s_matmul_cxt = functions->fullmat.data;
    }

    scalar_reduction_f = functions->scalar_redFun;
    complex_reduction_f = functions->complex_redFun;
#ifdef TIMER
    s_cputimef = &cputime;
#endif
    error = run_arnoldiabs(n_eigs, n_extend, tolerance, e_vecs, (lcomplex*)e_vals, maxIter, size, nMulti, stride, mode);
    if (error)
        goto cleanup;
    for (i = 0; i < n_eigs + n_extend; i++){
        if (i < n_eigs)
            results[i] = e_vals[i];
        if (rvecs && i < n_eigs)
            rvecs[i] = e_vecs[i];
        else
            free_fieldtype(e_vecs[i]);
        e_vecs[i] = NULL;
    }
cleanup:
    if (e_vecs){
        for (i = 0; i < n_eigs + n_extend; i++){
            if (e_vecs[i])
                free_fieldtype(e_vecs[i]);
        }
        s_freef(e_vecs);
    }
    if (e_vals)
        s_freef(e_vals);
    return error;
}




