/*
 * dlanczos.c
 *
 *  Created on: 1.7.2014
 *      Author: Teemu Rantalaiho
 *
 *
 *  Copyright 2014 Teemu Rantalaiho
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


#define RADIX_SIZE 64

// For normal complex types
typedef struct fieldtype_s
{
   double re;
} fieldtype;
typedef struct fieldentry_s
{
   double re;
} fieldentry;

#define MANGLE(X) dlanczos_##X
#define USE_LANCZOS
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
}
__device__
void set_fieldEntry(fieldtype* f, int i, const fieldentry* entry, int multiIdx, int stride){
    fieldtype tmp;
    tmp.re = entry->re;
    f[i+multiIdx*stride] = tmp;
}
__device__
lcomplex fieldEntry_dot(const fieldentry* a, const fieldentry* b){
    lcomplex res;
    // (z1*) * z2 = (x1 - iy1)*(x2 + iy2) = x1x2 + y1y2 + i(x1y2 - y1x2)
    res.real = a->re * b->re;
    res.imag = 0.0;
    return res;
}
__device__
radix fieldEntry_rdot(const fieldentry* a, const fieldentry* b){
    return a->re * b->re;
}
// dst = scalar * a
__device__
void fieldEntry_scalar_mult(const fieldentry* a, radix scalar, fieldentry* dst ){
    dst->re = scalar * a->re;
}
// dst = a + scalar * b
__device__
void fieldEntry_scalar_madd(const fieldentry * restrict a, radix scalar, const fieldentry * restrict b, fieldentry * restrict dst ){
    dst->re = a->re + scalar * b->re;
}
// dst = scalar * a
__device__
void fieldEntry_complex_mult(const fieldentry* a, lcomplex scalar, fieldentry* dst ){
    dst->re = scalar.real * a->re;
}
// dst = a + scalar * b
__device__
void fieldEntry_complex_madd(const fieldentry * restrict a, lcomplex scalar, const fieldentry * restrict b, fieldentry * restrict dst ){
    dst->re = a->re + scalar.real * b->re;
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
    }
    set_fieldEntry(in.dst, i, &x, multiIdx, in.stride);
}
PARALLEL_KERNEL_END()

static int s_size = 0;

#ifdef CUDA
#include "cuda_matmul.h"

template <typename INDEXTYPE>
struct ddMatMulFun {
    INDEXTYPE stride;
    __device__
    double operator()( const double* data, INDEXTYPE x, INDEXTYPE y, double* src, int srcx){
        double m_ij = data[y*stride + x];
        double res;
        double a = src[srcx];
        res = m_ij * a;
        return res;
    }
};

struct ddSumFun {
    __device__
    double operator()( double a, double b){

        return a+b;
    }
};
template <typename INDEXTYPE, typename ARRAYENTRYT>
struct ddStoreFun {
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
    double* m   = (double*)mat->data;
    double* x0  = (double*)src;
    double* y   = (double*)dst;
#ifndef CUDA
    // Trivial implementation - just for backup, replace at will with sgemv
    int i;
    int hopstride = mat->stride - size;
    double* lim = x0 + size;
    for (i = 0; i < size; i++){
        double* x = x0;
        double res;
        res = 0.0;
        while (x < lim){
            double a = *m++;
            double b = *x++;
            res += a * b;
        }
        *y++ = res;
        m += hopstride;
    }
#else
    ddMatMulFun<int> matmulf;
    ddStoreFun<int, double> storef;
    ddSumFun dsumf;
    matmulf.stride = mat->stride;
    error = (int)callFullMatMul<double>(m, matmulf, dsumf, storef, size, size, x0, y, false, 0, true);
#endif
    return error;
}

// Note - with a normal vector of complex numbers, use nMulti = 1, stride = 0
int run_dlanczos(
        dcomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride,
        int n_eigs, int n_extend, double tolerance, int* maxIter,
        const arnoldi_abs_int* functions, arnmode mode)
{
    int error = 0;

    fieldtype** e_vecs = (fieldtype**)s_mallocf(sizeof(fieldtype*) * (n_eigs + n_extend));
    dcomplex_t* e_vals = (dcomplex_t*)s_mallocf(sizeof(dcomplex_t) * (n_eigs + n_extend));
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




