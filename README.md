Cuda-Arnoldi
============

Parallel multi-CPU/GPU(CUDA)-implementation of the Implicitly Restarted Arnoldi Method
   by Teemu Rantalaiho, David Weir and Joni Suorsa (2011-13)

What is it
==========

  * An implementation in software of a mathematical algorithm to find eigenvalues
	and vectors of a large, possibly nonsymmetrical complex matrix. Based on the
	implicitly restarted Arnoldi method with deflation.

  * Written in C/C++ it exposes two levels of application programming interfaces:
	a high level interface which operates directly on vectors of complex numbers
	and a lower level interface, which can with very modest effort be made
	accommodate practically any kind of linear operators.

  * Provided as a couple of header-files and two source-files for the high level APIs:
       - `arnoldi_generic.h`- The actual arnoldi-method implementation,
								with all matrix-vector operations abstracted
       - `apar_defs.h`      - Implementations of the parallel loops/kernels used by
								arnoldi_generic.h and (c/z)arnoldi.c
       - `arnoldi_test.c`   - A very simple test-case developed using carnoldi.c
       - `arnoldi.h`        - Function prototypes for functions defined in carnoldi.c and
								zarnoldi.c as well as common type definitions
       - `carnoldi.c`       - Single precision high-level arnoldi method for complex vectors
       - `zarnoldi.c`       - Double precision high-level arnoldi method for complex vectors
       - `cuda_reduce.h`    - CUDA implementations for reductions used by apar_defs.h
       - `cuda_matmul.h`    - CUDA implementations for full matrix-vector multiplication
								used by apar_defs.h

How to use it
=============

  * See arnoldi_test.c for example usage and compilation instructions.
  * NOTE: Uses `zgeev()` or `cgeev()` from LAPACK-library for the small dense matrix operations
  * Full matrix-vector multiplication functions provided. Sparse versions to be implemented
	by the user and provided to the Arnoldi implementation by function pointers.

  * Multiple CPU-cores and multiple GPUs supported by MPI/other similar library by having
	the user provide functions that perform reductions of complex and floating point
	numbers across all nodes. See "reverse communication interface": `'arnoldi_abs_int'` in
	arnoldi.h for the high-level API and if you are using the low-level API you should assign
	the static variables `'scalar_reduction_f'` and `'complex_reduction_f'` appropriately.
	Other than this, use MPI (or equivalent) normally: each node handles its part of the
	problem and the Arnoldi code is otherwise oblivious to multiple nodes, except that it
	calls the reduction-functions to add together the results of the vector dot-products.
	(Except for the reductions, all inter-node communication needed is defined in the
	user-supplied matrix-vector multiplication function making the method very easily
	parallelizable in the conventional coarse-grained way.)

  * The software supports 4 modes: Largest magnitude, Largest real part, Smallest magnitude
	and smallest real part, but this only affects the sorting criteria. Advanced modes are
	left to the end-user (shift-invert, polynomial acceleration etc.
	See for example http://www.ecse.rpi.edu/~rjradke/papers/radkemathesis.pdf)

  * High-level API for CPU-implementation:
      - Compile carnoldi.c / zarnoldi.c with a C-compiler linking in your favorite
			lapack library (-llapack)
      - include `"arnoldi.h"` in your source file and call run_carnoldi()/run_zarnoldi()
		according to APIs defined in arnoldi.h

  * High-level API for CUDA-implementation:
      - Compile `"carnoldi.c / zarnoldi.c"` with `"nvcc -DCUDA -x cu"` and link in your
			favorite lapack library (`-llapack`)
      - include `"arnoldi.h"` in your source file and call `run_carnoldi()/run_zarnoldi()`
			according to API defined in arnoldi.h
      - Notice that the API assumes that the matrices and vectors are in device memory,
			also when using your own matrix-vector multiplication, the source and destination
			vectors point to device memory (of course)

  * Low-level API:
	- Define the vector type: `"fieldtype"` and the
		field-entry type `"fieldentry"` (one entry in a vector)
            * The algorithm works by calling the get/set_fieldEntry() functions through
				a one-dimensional index running from 0 to 'size-1'
				(and multiIdx from 0 to 'nMulti-1' providing a simple two-dimensional index)
	- `#define RADIX_SIZE 32 or 64`
    - include `"arnoldi_generic.h"`
    - Implement the following functions to define how `"fieldtype"` and `"fieldentry"` work:
        * `__host__ fieldtype* new_fieldtype(int size, int nMulti);`
        * `__host__ void free_fieldtype(fieldtype* field);`
        * `__device__ void get_fieldEntry
					(const fieldtype* f, int i, fieldentry* result, int multiIdx, int stride);`
        * `__device__ void set_fieldEntry
					(fieldtype* f, int i, const fieldentry* entry, int multiIdx, int stride);`
                  NOTE: `"multiIdx"` and `"stride"` enable finer-grained parallelization with
						SOA-data when using CUDA (can also be used for other purposes)
						`"stride"` is passed in through run_arnoldiabs() and is user-defined and
						`"multiIdx"` is an index that goes through {0,..., nMulti-1},
						where nMulti is passed by the user in `run_arnoldiabs()` - typically then
						`nMulti=1` and `stride=0` for simple complex vectors.
                        (These help if data is provided in a non-continuous manner, for example
						due to halo-sites in a multi-node configuration)
        * `__device__ lcomplex fieldEntry_dot(const fieldentry* a, const fieldentry* b);`
        * `__device__ radix fieldEntry_rdot(const fieldentry* a, const fieldentry* b);`
        * `__device__ void fieldEntry_scalar_mult
					(const fieldentry* a, radix scalar, fieldentry* dst);`
        * `__device__ void fieldEntry_scalar_madd
					(const fieldentry * a, radix scalar, const fieldentry * b, fieldentry * dst);`
        * `__device__ void fieldEntry_complex_mult
					(const fieldentry* a, lcomplex scalar, fieldentry* dst );`
        * `__device__ void fieldEntry_complex_madd
					(const fieldentry * a, lcomplex scalar, const fieldentry * b, fieldentry * dst);`
    - Set the following pointers and function pointers defined in arnoldi_generic.h
        * `"s_matmul_cxt"` - context for matrix multiplication function
        * `"s_mulf"` function that performs the matrix-vector multiplication
        * Optionally set `"scalar_reduction_f"` and `"complex_reduction_f"`
			to support MPI-parallelization (each MPI process handles a part of the vector)
        * Finally call `run_arnoldiabs()` - NOTE: the first `eig_vec[0]` in the call defines
			the initial vector (which will be normalized)
     - Compile normally
     - Compare with "zarnoldi.c" or "carnoldi.c"

Performance
===========

  * Preliminary studies would seem to indicate mostly similar convergence as the popular
		library 'arpack' wrt. number of Arnoldi restart iterations needed with some
		variation depending on case, but we have tested a very limited number of cases so far.
  * The code is mostly memory bandwidth bound which means that performance is what can be
		expected by the memory bandwidth of the used architecture.
  * With a Tesla K20m (ECC on - 175GB/s mem bw.) we achieve about 18.5x the performance
		of arpack++ on a single core of a Xeon X5650 @ 2.67GHz (32GB/s) on a 786432 sized
		system with a sparse (QCD) matrix with about 6 percent of the time spent in
		matrix-vector multiplies (on the GPU) with the same amount of reorthogonalizations
		done by both codes. On this use-case the GPU code achieved 146 Gbytes/s,
		which is 83 percent of theoretical peak memory bandwidth.
  * Scaling, measured on 1,2,4 and 8 Tesla M2050s (ECC on), shows that the algorithm scales
		well as long as the input-vectors are long enough to fill the GPUs. The method was
		seen to scale to vector-sizes as small as  196 000 complex numbers, which means 14000
		complex numbers for one streaming multiprocessor of the Tesla M2050.
		(In the scaling test the time taken in the user-defined matrix-vector multiplications
		 is taken out of the results)
  * The raw performance difference between our code on a single CPU core and arpack++
        seems to be around 36% in favor of arpack++ with the deflation part taking most of
		the extra time (32 percent of the whole method runtime - in this particular case
		convergence was quite fast making the cost related to deflation relatively high -
		also it should be noted that our optimizations have been targeted towards the GPU
		leaving further improvements on the CPU side possible).


Issues and Future Work
======================

  * Initial release of software with a modest amount of testing performed, which means
		that some issues are likely to come up.
  * No adaptive reorthogonalization / iterative refinement strategy in arnoldi-iteration step
       (These would be quite simple to implement but the "normal" case seems to be that one
		re-orthogonalization is required but no more)
  * Error handling API very lightly tested
  * No invariant subspace detection: an invariant subspace of the matrix has been found
		if the residual vector in the Arnoldi iteration goes to zero and therefore the
		Krylov subspace cannot be further expanded. In this case a possible solution would
		be to select a new starting vector and start again.
  * Code is very lightly commented, but most of it should be easy to read.
  * deflate() should be optimized wrt. high number of requested eigenvalues (~ 100 and up).
		The method does 3 matrix-matrix multiplications in a loop on the "small" matrices, 
		which can start to show up when the small matrix is not so small wrt. problem size.
		One obvious way to handle this is to do it on the GPU.
  * Suggestions how to improve the software will be greatly appreciated.
		(Email: teemu.rantalaiho at helsinki.fi)


