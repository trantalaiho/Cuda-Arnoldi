/*
 * arnoldi.h
 *
 *  Created on: 7.8.2013
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

#ifndef ARNOLDI_H_
#define ARNOLDI_H_

// Defines sorting criteria (nothing more) for the algorithm
// Use the "Reverse Communication Interface" to get more options
// by redefining your matrix
// Also note that using LM mode should converge fastest due
// to the power-iteration nature of the method
typedef enum arnmode_s{
    arnmode_LM = 1, // Largest magnitude
    arnmode_LR,     // Largest real part
    arnmode_SM,     // Smallest magnitude
    arnmode_SR,     // Smallest real part

    arnmode_force32bit = 0x7FFFFFF // Force size of type to be at least 32 bits
} arnmode;


typedef struct scomplex_s {
    float re;
    float im;
} scomplex_t;

typedef struct dcomplex_s {
    double re;
    double im;
} dcomplex_t;

/* Compute y <- Ax - matctx is an arbitrary user context needed in matrix-vector multiply */
typedef void (*smatvecmult) (void* matctx, float* src, float* dst);
typedef void (*dmatvecmult) (void* matctx, double* src, double* dst);
typedef void (*cmatvecmult) (void* matctx, scomplex_t* src, scomplex_t* dst);
typedef void (*zmatvecmult) (void* matctx, dcomplex_t* src, dcomplex_t* dst);
typedef void (*amatvecmult) (void* matctx, void* src, void* dst);

/* Do reduction of val across all nodes - for single-node support, just leave this NULL */
typedef void (*mpi_reductiont) (void* val);

typedef void* (*fieldAllocT) (int size, int nMulti);
typedef void (*fieldFreeT) (void* fieldPtr);


/* Matrix elements stored as: m_ij = data[j + stride * i] */
typedef struct fullmat_s{ void* data; int stride;} fullmat_t;
typedef struct arnoldi_abs_int_s{
    fullmat_t  fullmat;             // Matrix data, if mvec_mul not given, then data must contain data as expected
    amatvecmult mvecmulFun;         // Matrix-vector multiplication function - optional, if set then fullmat.data given
                                    //                            as context to the function to pass in arbitrary data
    fieldAllocT allocFieldFun;      // Function pointer to allocate a field - optional
    fieldFreeT  freeFieldFun;       // Function pointer to free a field - optional
    mpi_reductiont scalar_redFun;   // Scalar reduction function - optional
    mpi_reductiont complex_redFun;  // Complex reduction function - optional
    void* reserve1; // For futureproofing
    void* reserve2;
} arnoldi_abs_int;

#ifndef EXTERN_C_BEGIN
#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif
#endif



// init_vec contains the initial field vector (type depends on fieldAllocT) for Arnoldi iteration (optional)
// rvecs should be a pointer to an array of n_eigs pointers or NULL - if not NULL, will contain the right eigenvectors corresponding to results
// maxIter contains the number of iterations used
EXTERN_C_BEGIN
int run_sarnoldi(scomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
int run_darnoldi(dcomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
int run_carnoldi(scomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
int run_zarnoldi(dcomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);

int run_slanczos(scomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
int run_dlanczos(dcomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
int run_clanczos(scomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
int run_zlanczos(dcomplex_t* results, const void* init_vec, void** rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, const arnoldi_abs_int* functions, arnmode mode);
EXTERN_C_END

#endif /* ARNOLDI_H_ */
