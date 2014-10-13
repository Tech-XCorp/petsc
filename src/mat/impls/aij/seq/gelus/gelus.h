#if !defined(__GELUSMATIMPL)
#define __GELUSMATIMPL

#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#include <cuGelus.h>

#include <algorithm>
#include <vector>

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)  
#define cugelus_ilu_solve          cugelusCcsrilu_solve
#define cugelus_ilu_solve_analysis cugelusCcsrilu_solve_analysis
#define cugelus_ic_solve           cugelusCcsric_solve
#define cugelus_ic_solve_analysis  cugelusCcsric_solve_analysis
typedef gelusFloatComplex GELUSScalar;
#elif defined(PETSC_USE_REAL_DOUBLE)
#define cugelus_ilu_solve          cugelusZcsrilu_solve
#define cugelus_ilu_solve_analysis cugelusZcsrilu_solve_analysis
#define cugelus_ic_solve           cugelusZcsric_solve
#define cugelus_ic_solve_analysis  cugelusZcsric_solve_analysis
typedef gelusDoubleComplex GELUSScalar;
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)  
#define cugelus_ilu_solve          cugelusScsrilu_solve
#define cugelus_ilu_solve_analysis cugelusScsrilu_solve_analysis
#define cugelus_ic_solve           cugelusScsric_solve
#define cugelus_ic_solve_analysis  cugelusScsric_solve_analysis
typedef float GELUSScalar;
#elif defined(PETSC_USE_REAL_DOUBLE)
#define cugelus_ilu_solve          cugelusDcsrilu_solve
#define cugelus_ilu_solve_analysis cugelusDcsrilu_solve_analysis
#define cugelus_ic_solve           cugelusDcsric_solve
#define cugelus_ic_solve_analysis  cugelusDcsric_solve_analysis
typedef double GELUSScalar;
#endif
#endif

#if defined(THRUSTINTARRAY32)
#undef THRUSTINTARRAY32
#endif
#define THRUSTINTARRAY32 thrust::device_vector<int>
#if defined(THRUSTINTARRAY)
#undef THRUSTINTARRAY
#endif
#define THRUSTINTARRAY thrust::device_vector<PetscInt>
#if defined(THRUSTARRAY)
#undef THRUSTARRAY
#endif
#define THRUSTARRAY thrust::device_vector<PetscScalar>

typedef enum {GELUS_ILU,GELUS_IC,GELUS_SOR} GelusPreconditionerType;

/* This is struct holding the relevant data needed to do a MatSolve */
struct Mat_GelusSolveStruct {
  /* Data needed for triangular solve */
  void*                   solveData;
  gelusSolveDescription_t solveDescription;
  GelusPreconditionerType precType;
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and
 any indices used in a reordering */
struct Mat_GelusFactors {
  Mat_GelusSolveStruct             *factorPtr;          /* pointer for factored matrix on GPU */
  Mat_GelusSolveStruct             *factorPtrTranspose; /* pointer for factored transposed matrix on GPU */
  THRUSTINTARRAY                   *rpermIndices;       /* indices used for any reordering */
  THRUSTINTARRAY                   *cpermIndices;       /* indices used for any reordering */
  THRUSTARRAY                      *workVector;
  PetscInt                         nnz;                 /* number of nonzeros ... need this for accurate logging between ICC and ILU */
  gelusStorageFormat_t             storage;
  gelusOptimizationLevel_t         optimizationLevel;
};

#endif
