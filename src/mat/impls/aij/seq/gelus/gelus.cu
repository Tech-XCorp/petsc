/*
   Defines solvers using gelus.
*/

#include "petscconf.h"
#include "../src/mat/impls/aij/seq/aij.h"
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"
#include <cuGelus.h>
#include <../src/mat/impls/aij/seq/gelus/gelus.h>
#include <stdio.h>

#ifdef PETSC_USE_LOG
#define PETSC_GELUS_USE_LOG
#endif

#if defined PETSC_GELUS_USE_LOG
static int cugelusLogRegistered = 0;
static PetscLogEvent cugelusICAnalze;
static PetscLogEvent cugelusILAnalze;
static PetscLogEvent cugelusICSolve;
static PetscLogEvent cugelusILSolve;
static PetscLogEvent cugelusICDestr;
static PetscLogEvent cugelusILDestr;
#endif

#undef VecType

const char *const MatGELUSStorageFormats[] = {"CSR","ELL","HYB","MatGELUSStorageFormat","MAT_GELUS_",0};

static PetscErrorCode MatICCFactorSymbolic_GELUS(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorSymbolic_GELUS(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorNumeric_GELUS(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatILUFactorSymbolic_GELUS(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorSymbolic_GELUS(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorNumeric_GELUS(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatSolve_GELUS(Mat,Vec,Vec);
static PetscErrorCode MatSolve_GELUS_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_GELUS(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_GELUS_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSetFromOptions_GELUS(Mat);
static PetscErrorCode MatDestroy_GELUS(Mat);

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_gelus"
PetscErrorCode MatFactorGetSolverPackage_seqaij_gelus(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERGELUS;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERGELUS = "gelus" - A solver package with GPU accelerated
  triangular solves using cuda.  This class does NOT support direct
  solver operations.

  ./configure --download-gelus to install PETSc to use gelus

  Consult GELUS documentation for more information about the Control parameters
  which correspond to the options database keys below.

  Options Database Keys:
+ -mat_gelus_storage_format <CSR>                  - (choose one of) CSR, ELL, or HYB
. -mat_gelus_optimization_strategy <BOTH>          - (choose one of) SETUP, SOLVE, or BOTH
- -mat_gelus_kernel_consolidation_amount <DYNAMIC> - (choose one of) NONE, TWO, THREE, ..., TEN, DYNAMIC, ALL

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage
M*/

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqaij_gelus"
PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_gelus(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode    ierr;
  PetscInt          n=A->rmap->n;
  Mat_GelusFactors* factorData;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(*B);CHKERRQ(ierr);
  factorData = new Mat_GelusFactors;
  factorData->factorPtr          = 0;
  factorData->factorPtrTranspose = 0;
  factorData->rpermIndices       = 0;
  factorData->cpermIndices       = 0;
  factorData->workVector         = 0;
  factorData->nnz                = 0;
  ((Mat)*B)->spptr               = factorData;
  ((Mat)*B)->ops->destroy        = MatDestroy_GELUS;
  ((Mat)*B)->ops->setfromoptions = MatSetFromOptions_GELUS;

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetBlockSizes(*B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_GELUS;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_GELUS;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_GELUS;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_GELUS;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for GELUS Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_seqaij_gelus);CHKERRQ(ierr);


#if defined PETSC_GELUS_USE_LOG
  if (cugelusLogRegistered == 0) {
    ierr = PetscLogEventRegister("cugelusICAnalze", MAT_CLASSID, &cugelusICAnalze);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("cugelusILAnalze", MAT_CLASSID, &cugelusILAnalze);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("cugelusICSolve ", MAT_CLASSID, &cugelusICSolve );CHKERRQ(ierr);
    ierr = PetscLogEventRegister("cugelusILSolve ", MAT_CLASSID, &cugelusILSolve );CHKERRQ(ierr);
    ierr = PetscLogEventRegister("cugelusICDestr ", MAT_CLASSID, &cugelusICDestr );CHKERRQ(ierr);
    ierr = PetscLogEventRegister("cugelusILDestr ", MAT_CLASSID, &cugelusILDestr );CHKERRQ(ierr);
    cugelusLogRegistered = 1;
  }
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatILUFactorSymbolic_GELUS"
static PetscErrorCode MatILUFactorSymbolic_GELUS(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_GELUS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_GELUS"
static PetscErrorCode MatLUFactorSymbolic_GELUS(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_GELUS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatICCFactorSymbolic_GELUS"
static PetscErrorCode MatICCFactorSymbolic_GELUS(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatICCFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_GELUS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_GELUS"
static PetscErrorCode MatCholeskyFactorSymbolic_GELUS(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_GELUS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGELUSBuildILUCSRMatrix"
static PetscErrorCode MatGELUSBuildILUCSRMatrix(Mat A, gelusOperation_t OP)
{
  Mat_SeqAIJ              *a = (Mat_SeqAIJ*)A->data;
  PetscInt                n = A->rmap->n;
  Mat_GelusFactors        *gelusFactors = (Mat_GelusFactors*)A->spptr;
  const PetscInt          *ai = a->i,*aj = a->j,*vi,*adiag=a->diag;
  PetscInt                *lowerNNZPerRow,*upperNNZPerRow,*lowerRowOffsets,*upperRowOffsets;
  const MatScalar         *aa = a->a,*v;
  PetscInt                *Ai, *Aj;
  PetscScalar             *AA;
  PetscInt                i, nz, nnz, nzLower, nzUpper;
  gelusSolveDescription_t solveDesc;
  cugelusIluSolveData_t   h;
  gelusStatus_t           gStat=GELUS_STATUS_SUCCESS;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  nzLower=ai[n]-ai[1];
  nzUpper=adiag[0]-adiag[n];
  nnz=nzUpper+nzLower;

  ierr = PetscMalloc((n+1)*sizeof(PetscInt),(void**) &Ai);CHKERRQ(ierr);
  ierr = PetscMalloc(nnz*sizeof(PetscInt),(void**) &Aj);CHKERRQ(ierr);
  ierr = PetscMalloc(nnz*sizeof(PetscScalar),(void**) &AA);CHKERRQ(ierr);

  ierr = PetscMalloc(n*sizeof(PetscInt),(void**)&lowerNNZPerRow);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),(void**)&upperNNZPerRow);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),(void**)&lowerRowOffsets);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),(void**)&upperRowOffsets);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    nz = ai[i+1] - ai[i];
    lowerNNZPerRow[i]=nz;
  }
  for (i=n-1; i>=0; i--) {
    nz = adiag[i] - adiag[i+1];
    upperNNZPerRow[i]=nz;
  }
  lowerRowOffsets[0]=0;
  upperRowOffsets[0]=lowerNNZPerRow[0];
  for (i=1;i<n;++i) {
    lowerRowOffsets[i]=upperRowOffsets[i-1]+upperNNZPerRow[i-1];
    upperRowOffsets[i]=lowerRowOffsets[i]+lowerNNZPerRow[i];
  }

  Ai[0] = (PetscInt) 0;
  Ai[n] = nnz;
  v     = aa;
  vi    = aj;
  for (i=1; i<n; i++) {
    nz = lowerNNZPerRow[i];
    Ai[i]    = lowerRowOffsets[i];
    ierr = PetscMemcpy(&(Aj[lowerRowOffsets[i]]), vi, nz*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(&(AA[lowerRowOffsets[i]]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);
    v  += nz;
    vi += nz;
  }
  for (i=n-1; i>=0; i--) {
    v  = aa + adiag[i+1] + 1;
    vi = aj + adiag[i+1] + 1;
    /* number of elements NOT on the diagonal */
    nz = adiag[i] - adiag[i+1]-1;
    Aj[upperRowOffsets[i]] = (PetscInt) i;
    AA[upperRowOffsets[i]] = 1./v[nz];
    ierr = PetscMemcpy(&(Aj[upperRowOffsets[i]+1]), vi, (upperNNZPerRow[i]-1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(&(AA[upperRowOffsets[i]+1]), v, (upperNNZPerRow[i]-1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  gStat = gelusCreateSolveDescr(&solveDesc);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }

  gStat = gelusSetSolveIndexBase(solveDesc, GELUS_INDEX_BASE_ZERO);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetSolveFillMode(solveDesc, GELUS_FILL_MODE_FULL);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetSolveOperation(solveDesc, OP);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetSolveStorageFormat(solveDesc, gelusFactors->storage);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetOptimizationLevel(solveDesc, gelusFactors->optimizationLevel);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }

  gStat = cugelusCreateIluSolveData(&h);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILAnalze,A,0,0,0);CHKERRQ(ierr);
#endif
  gStat = cugelus_ilu_solve_analysis(n,solveDesc,Ai,Aj,(GELUSScalar*)AA,h);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILAnalze,A,0,0,0);CHKERRQ(ierr);
#endif
  if (OP == GELUS_OPERATION_NON_TRANSPOSE) {
    Mat_GelusSolveStruct *factorPtr;
    factorPtr                   = new Mat_GelusSolveStruct;
    factorPtr->precType         = GELUS_ILU;
    factorPtr->solveDescription = solveDesc;
    factorPtr->solveData        = h;
    gelusFactors->factorPtr     = factorPtr;
  } else {
    Mat_GelusSolveStruct *factorPtrTranspose;
    factorPtrTranspose                   = new Mat_GelusSolveStruct;
    factorPtrTranspose->precType         = GELUS_ILU;
    factorPtrTranspose->solveDescription = solveDesc;
    factorPtrTranspose->solveData        = h;
    gelusFactors->factorPtrTranspose     = factorPtrTranspose;
  }

  /* free temporary data */
  ierr = PetscFree(Ai);CHKERRQ(ierr);
  ierr = PetscFree(Aj);CHKERRQ(ierr);
  ierr = PetscFree(AA);CHKERRQ(ierr);
  ierr = PetscFree(lowerNNZPerRow);CHKERRQ(ierr);
  ierr = PetscFree(upperNNZPerRow);CHKERRQ(ierr);
  ierr = PetscFree(lowerRowOffsets);CHKERRQ(ierr);
  ierr = PetscFree(upperRowOffsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGELUSILUAnalysisAndCopyToGPU"
static PetscErrorCode MatGELUSILUAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode   ierr;
  PetscBool        row_identity,col_identity;
  const PetscInt   *r,*c;
  Mat_SeqAIJ       *a            = (Mat_SeqAIJ*)A->data;
  Mat_GelusFactors *gelusFactors = (Mat_GelusFactors*)A->spptr;
  IS               isrow         = a->row;
  IS               iscol         = a->icol;
  PetscInt         n             = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatGELUSBuildILUCSRMatrix(A, GELUS_OPERATION_NON_TRANSPOSE);CHKERRQ(ierr);

  gelusFactors->workVector = new THRUSTARRAY;
  gelusFactors->workVector->resize(n);
  gelusFactors->nnz=a->nz;

  /*lower triangular indices */
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  if (!row_identity) {
    gelusFactors->rpermIndices = new THRUSTINTARRAY(n);
    gelusFactors->rpermIndices->assign(r, r+n);
  }
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  /*upper triangular indices */
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!col_identity) {
    gelusFactors->cpermIndices = new THRUSTINTARRAY(n);
    gelusFactors->cpermIndices->assign(c, c+n);
  }
  ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_GELUS"
static PetscErrorCode MatLUFactorNumeric_GELUS(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b    = (Mat_SeqAIJ*)B->data;
  IS             isrow = b->row;
  IS             iscol = b->col;
  PetscBool      row_identity,col_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    B->ops->solve = MatSolve_GELUS_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_GELUS_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_GELUS;
    B->ops->solvetranspose = MatSolveTranspose_GELUS;
  }

  ierr = MatGELUSILUAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatGELUSBuildICCCSRMatrix"
static PetscErrorCode MatGELUSBuildICCCSRMatrix(Mat A, gelusOperation_t OP)
{
  Mat_SeqAIJ              *a = (Mat_SeqAIJ*)A->data;
  Mat_GelusFactors        *gelusFactors = (Mat_GelusFactors*)A->spptr;
  gelusSolveDescription_t solveDesc;
  cugelusIcSolveData_t    h;
  gelusStatus_t           gStat = GELUS_STATUS_SUCCESS;
  PetscErrorCode          ierr;
  PetscInt                *Ai, *Aj;
  PetscScalar             *AA;
  PetscInt                nnz = a->nz,n = A->rmap->n,i,offset,nz,j;
  Mat_SeqSBAIJ            *b = (Mat_SeqSBAIJ*)A->data;
  const PetscInt          *ai = b->i,*aj = b->j,*vj;
  const MatScalar         *aa = b->a,*v;
  MatScalar               diagonal;

  PetscFunctionBegin;
  /* Allocate Space for the upper triangular matrix */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),(void**) &Ai);CHKERRQ(ierr);
  ierr = PetscMalloc(nnz*sizeof(PetscInt),(void**) &Aj);CHKERRQ(ierr);
  ierr = PetscMalloc(nnz*sizeof(PetscScalar),(void**) &AA);CHKERRQ(ierr);

  /* Fill the upper triangular matrix */
  Ai[0]=(PetscInt) 0;
  Ai[n]=nnz;
  offset = 0;
  for (i=0; i<n; i++) {
    /* set the pointers */
    v  = aa + ai[i];
    vj = aj + ai[i];
    nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */

    /* first, set the diagonal elements */
    Aj[offset] = (PetscInt) i;
#if defined(PETSC_USE_COMPLEX)
    diagonal = PetscSqrtComplex(v[nz]);
#else
    diagonal = PetscSqrtReal(v[nz]);
#endif
    AA[offset] = 1.0/diagonal;
    Ai[i]      = offset;

    offset+=1;
    if (nz>0) {
      ierr = PetscMemcpy(&(Aj[offset]), vj, nz*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(&(AA[offset]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=offset; j<offset+nz; j++) {
        AA[j] = -AA[j]/diagonal;
      }
      offset+=nz;
    }
  }

  gStat = gelusCreateSolveDescr(&solveDesc);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }

  gStat = gelusSetSolveIndexBase(solveDesc, GELUS_INDEX_BASE_ZERO);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetSolveFillMode(solveDesc, GELUS_FILL_MODE_UPPER);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetSolveOperation(solveDesc, OP);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetSolveStorageFormat(solveDesc, gelusFactors->storage);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
  gStat = gelusSetOptimizationLevel(solveDesc, gelusFactors->optimizationLevel);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }

  gStat = cugelusCreateIcSolveData(&h);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICAnalze,A,0,0,0);CHKERRQ(ierr);
#endif
  gStat = cugelus_ic_solve_analysis(n,solveDesc,Ai,Aj,(GELUSScalar*)AA,h);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICAnalze,A,0,0,0);CHKERRQ(ierr);
#endif

  if (OP == GELUS_OPERATION_NON_TRANSPOSE) {
    Mat_GelusSolveStruct *factorPtr;
    factorPtr                   = new Mat_GelusSolveStruct;
    factorPtr->precType         = GELUS_IC;
    factorPtr->solveDescription = solveDesc;
    factorPtr->solveData        = h;
    gelusFactors->factorPtr     = factorPtr;
  } else {
    Mat_GelusSolveStruct *factorPtrTranspose;
    factorPtrTranspose                   = new Mat_GelusSolveStruct;
    factorPtrTranspose->precType         = GELUS_IC;
    factorPtrTranspose->solveDescription = solveDesc;
    factorPtrTranspose->solveData        = h;
    gelusFactors->factorPtrTranspose     = factorPtrTranspose;
  }


  /* free temporary data */
  ierr = PetscFree(Ai);CHKERRQ(ierr);
  ierr = PetscFree(Aj);CHKERRQ(ierr);
  ierr = PetscFree(AA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGELUSICCAnalysisAndCopyToGPU"
static PetscErrorCode MatGELUSICCAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *a            = (Mat_SeqAIJ*)A->data;
  Mat_GelusFactors *gelusFactors = (Mat_GelusFactors*)A->spptr;
  IS               ip = a->row;
  const PetscInt   *rip;
  PetscBool        perm_identity;
  PetscInt         n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatGELUSBuildICCCSRMatrix(A, GELUS_OPERATION_NON_TRANSPOSE);CHKERRQ(ierr);
  gelusFactors->workVector = new THRUSTARRAY;
  gelusFactors->workVector->resize(n);
  gelusFactors->nnz=(a->nz-n)*2 + n;

  /*lower triangular indices */
  ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) {
    gelusFactors->rpermIndices = new THRUSTINTARRAY(n);
    gelusFactors->rpermIndices->assign(rip, rip+n);
    gelusFactors->cpermIndices = new THRUSTINTARRAY(n);
    gelusFactors->cpermIndices->assign(rip, rip+n);
  }
  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_GELUS"
static PetscErrorCode MatCholeskyFactorNumeric_GELUS(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);

  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity) {
    B->ops->solve = MatSolve_GELUS_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_GELUS_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_GELUS;
    B->ops->solvetranspose = MatSolveTranspose_GELUS;
  }

  /* get the triangular factors */
  ierr = MatGELUSICCAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGELUSAnalyzeTransposeForSolve"
static PetscErrorCode MatGELUSAnalyzeTransposeForSolve(Mat A)
{
  Mat_GelusFactors     *gelusStruct = (Mat_GelusFactors*)A->spptr;
  Mat_GelusSolveStruct  *factorPtr = (Mat_GelusSolveStruct*)gelusStruct->factorPtr;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (factorPtr->precType == GELUS_ILU) {
    ierr = MatGELUSBuildILUCSRMatrix(A, GELUS_OPERATION_TRANSPOSE);CHKERRQ(ierr);
  } else {
    ierr = MatGELUSBuildICCCSRMatrix(A, GELUS_OPERATION_TRANSPOSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_GELUS"
static PetscErrorCode MatSolveTranspose_GELUS(Mat A,Vec bb,Vec xx)
{
  CUSPARRAY            *xGPU, *bGPU;
  Mat_GelusFactors     *gelusStruct = (Mat_GelusFactors*)A->spptr;
  Mat_GelusSolveStruct *factorPtrTranspose = (Mat_GelusSolveStruct*)gelusStruct->factorPtrTranspose;
  THRUSTARRAY          *tempGPU = (THRUSTARRAY*)gelusStruct->workVector;
  gelusStatus_t        gStat = GELUS_STATUS_SUCCESS;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!factorPtrTranspose) {
    ierr = MatGELUSAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    factorPtrTranspose = (Mat_GelusSolveStruct*)gelusStruct->factorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU->begin(), gelusStruct->rpermIndices->begin()),
      thrust::make_permutation_iterator(bGPU->end(), gelusStruct->rpermIndices->end()),
      xGPU->begin());

  /* Solve with GELUS */
  if (factorPtrTranspose->precType == GELUS_ILU) {
    cugelusIluSolveData_t h = (cugelusIluSolveData_t)factorPtrTranspose->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ilu_solve((GELUSScalar*)xGPU->data().get(),(GELUSScalar*)tempGPU->data().get(),h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  } else {
    cugelusIcSolveData_t h = (cugelusIcSolveData_t)factorPtrTranspose->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ic_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)tempGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  }

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::make_permutation_iterator(xGPU->begin(), gelusStruct->cpermIndices->begin()),
      thrust::make_permutation_iterator(xGPU->end(), gelusStruct->cpermIndices->end()),
      tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(tempGPU->begin(), tempGPU->end(), xGPU->begin());

  /* restore */
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);

  ierr = PetscLogFlops(2.0*gelusStruct->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_GELUS_NaturalOrdering"
static PetscErrorCode MatSolveTranspose_GELUS_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  CUSPARRAY            *xGPU,*bGPU;
  Mat_GelusFactors     *gelusStruct = (Mat_GelusFactors*)A->spptr;
  Mat_GelusSolveStruct *factorPtrTranspose = (Mat_GelusSolveStruct*)gelusStruct->factorPtrTranspose;
  gelusStatus_t        gStat = GELUS_STATUS_SUCCESS;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!factorPtrTranspose) {
    ierr = MatGELUSAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    factorPtrTranspose = (Mat_GelusSolveStruct*)gelusStruct->factorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* Solve with GELUS */
  if (factorPtrTranspose->precType == GELUS_ILU) {
    cugelusIluSolveData_t h = (cugelusIluSolveData_t)factorPtrTranspose->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ilu_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)bGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  } else {
    cugelusIcSolveData_t h = (cugelusIcSolveData_t)factorPtrTranspose->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ic_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)bGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  }

  /* restore */
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*gelusStruct->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_GELUS"
static PetscErrorCode MatSolve_GELUS(Mat A,Vec bb,Vec xx)
{
  CUSPARRAY             *xGPU,*bGPU;
  Mat_GelusFactors      *gelusStruct = (Mat_GelusFactors*)A->spptr;
  Mat_GelusSolveStruct  *factorPtr = (Mat_GelusSolveStruct*)gelusStruct->factorPtr;
  THRUSTARRAY           *tempGPU = (THRUSTARRAY*)gelusStruct->workVector;
  gelusStatus_t         gStat = GELUS_STATUS_SUCCESS;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU->begin(), gelusStruct->rpermIndices->begin()),
      thrust::make_permutation_iterator(bGPU->end(), gelusStruct->rpermIndices->end()),
      tempGPU->begin());

  /* Solve with GELUS */
  if (factorPtr->precType == GELUS_ILU) {
    cugelusIluSolveData_t h = (cugelusIluSolveData_t)factorPtr->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ilu_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)tempGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  } else {
    cugelusIcSolveData_t h = (cugelusIcSolveData_t)factorPtr->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ic_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)tempGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  }

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::make_permutation_iterator(xGPU->begin(), gelusStruct->cpermIndices->begin()),
      thrust::make_permutation_iterator(xGPU->end(), gelusStruct->cpermIndices->end()),
      tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(tempGPU->begin(), tempGPU->end(), xGPU->begin());

  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*gelusStruct->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_GELUS_NaturalOrdering"
static PetscErrorCode MatSolve_GELUS_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  CUSPARRAY             *xGPU,*bGPU;
  Mat_GelusFactors      *gelusStruct = (Mat_GelusFactors*)A->spptr;
  Mat_GelusSolveStruct  *factorPtr   = (Mat_GelusSolveStruct*)gelusStruct->factorPtr;
  gelusStatus_t         gStat = GELUS_STATUS_SUCCESS;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* Solve with GELUS */
  if (factorPtr->precType == GELUS_ILU) {
    cugelusIluSolveData_t h = (cugelusIluSolveData_t)factorPtr->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ilu_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)bGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  } else {
    cugelusIcSolveData_t h = (cugelusIcSolveData_t)factorPtr->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
    gStat = cugelus_ic_solve((GELUSScalar*)xGPU->data().get(), (GELUSScalar*)bGPU->data().get(), h);
    if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICSolve,A,0,0,0);CHKERRQ(ierr);
#endif
  }

  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*gelusStruct->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_GELUS"
static PetscErrorCode MatDestroy_GELUS(Mat A)
{
  Mat_GelusFactors      *gelusStruct = (Mat_GelusFactors*)A->spptr;
  Mat_GelusSolveStruct  *factorPtr   = (Mat_GelusSolveStruct*)gelusStruct->factorPtr;
  Mat_GelusSolveStruct  *factorPtrTranspose   = (Mat_GelusSolveStruct*)gelusStruct->factorPtrTranspose;
  gelusSolveDescription_t descr;
  PetscErrorCode        ierr;
  gelusStatus_t         gStat;

  PetscFunctionBegin;
  if (gelusStruct) {
    if (factorPtr) {
      /* destroy the solve data */
      if (factorPtr->precType==GELUS_ILU) {
        cugelusIluSolveData_t h = (cugelusIluSolveData_t)factorPtr->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILDestr,A,0,0,0);CHKERRQ(ierr);
#endif
        gStat = cugelusDestroyIluSolveData(h);
        if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILDestr,A,0,0,0);CHKERRQ(ierr);
#endif
      } else if (factorPtr->precType==GELUS_ILU) {
        cugelusIcSolveData_t h = (cugelusIcSolveData_t)factorPtr->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICDestr,A,0,0,0);CHKERRQ(ierr);
#endif
        gStat = cugelusDestroyIcSolveData(h);
        if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICDestr,A,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* destroy the solve description */
      descr = factorPtr->solveDescription;
      gStat = gelusDestroySolveDescr(descr);
      if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
    }
    delete (Mat_GelusSolveStruct*)factorPtr;
    factorPtr = 0;

    if (factorPtrTranspose) {
      /* destroy the solve data */
      if (factorPtrTranspose->precType==GELUS_ILU) {
        cugelusIluSolveData_t h = (cugelusIluSolveData_t)factorPtrTranspose->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusILDestr,A,0,0,0);CHKERRQ(ierr);
#endif
        gStat = cugelusDestroyIluSolveData(h);
        if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusILDestr,A,0,0,0);CHKERRQ(ierr);
#endif
      } else if (factorPtr->precType==GELUS_ILU) {
        cugelusIcSolveData_t h = (cugelusIcSolveData_t)factorPtrTranspose->solveData;
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventBegin(cugelusICDestr,A,0,0,0);CHKERRQ(ierr);
#endif
        gStat = cugelusDestroyIcSolveData(h);
        if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusICDestr,A,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* destroy the solve description */
      descr = factorPtrTranspose->solveDescription;
      gStat = gelusDestroySolveDescr(descr);
      if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
    }
    delete (Mat_GelusSolveStruct*)factorPtrTranspose;
    factorPtrTranspose = 0;

    if (gelusStruct->rpermIndices) delete (THRUSTINTARRAY*)gelusStruct->rpermIndices;
    if (gelusStruct->cpermIndices) delete (THRUSTINTARRAY*)gelusStruct->cpermIndices;
    if (gelusStruct->workVector) delete (THRUSTARRAY*)gelusStruct->workVector;
    delete (Mat_GelusFactors*)gelusStruct;
  }

  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;

  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetFromOptions_GELUS"
PetscErrorCode MatSetFromOptions_GELUS(Mat A)
{
  Mat_GelusFactors* factorData;
  const char        *storage[]={"CSR","ELL","HYB"};
  const char        *optimizationLevel[]={"OZERO","OONE","OTWO","OTHREE"};
  PetscBool         flg;
  PetscInt          idx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  factorData = (Mat_GelusFactors*)A->spptr;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"GELUS Options","Mat");CHKERRQ(ierr);
  /* Control parameters for the storage format */
  ierr = PetscOptionsEList("-mat_gelus_storage_format","Storage format : csr, ell, or hyb","None",storage,3,storage[0],&idx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (idx) {
    case 0: factorData->storage = GELUS_STORAGE_FORMAT_CSR; break;
    case 1: factorData->storage = GELUS_STORAGE_FORMAT_ELL; break;
    case 2: factorData->storage = GELUS_STORAGE_FORMAT_HYB; break;
    }
  } else {
    factorData->storage = GELUS_STORAGE_FORMAT_CSR;
  }

  /* Control parameters for the optimization strategy */
  ierr = PetscOptionsEList("-mat_gelus_optimization_level","Optimization level to be used by gelus","None",optimizationLevel,4,optimizationLevel[3],&idx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (idx) {
    case 0: factorData->optimizationLevel = GELUS_OPTIMIZATION_LEVEL_ZERO; break;
    case 1: factorData->optimizationLevel = GELUS_OPTIMIZATION_LEVEL_ONE; break;
    case 2: factorData->optimizationLevel = GELUS_OPTIMIZATION_LEVEL_TWO; break;
    case 3: factorData->optimizationLevel = GELUS_OPTIMIZATION_LEVEL_THREE; break;
    }
  } else {
    factorData->optimizationLevel = GELUS_OPTIMIZATION_LEVEL_THREE;
  }

  PetscOptionsEnd();
  PetscFunctionReturn(0);
}
