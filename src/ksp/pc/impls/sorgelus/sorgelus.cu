#include <petscsys.h>
#include <petsc-private/pcimpl.h>
#include "../src/mat/impls/aij/seq/aij.h"
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>
#include <cuGelus.h>

#ifdef PETSC_USE_LOG
#define PETSC_GELUS_USE_LOG
#endif

#if !defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)  
#define cugelus_csr_iteration_analysis cugelusScsrsor_iteration_analysis
#define GELUS_SCALAR_TYPE              float
#define cugelus_csrsor_iterate         cugelusScsrsor_iterate
#elif defined(PETSC_USE_REAL_DOUBLE)
#define cugelus_csr_iteration_analysis cugelusDcsrsor_iteration_analysis
#define GELUS_SCALAR_TYPE              double
#define cugelus_csrsor_iterate         cugelusDcsrsor_iterate
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)  
#define cugelus_csr_iteration_analysis cugelusCcsrsor_iteration_analysis
#define GELUS_SCALAR_TYPE              gelusFloatComplex
#define cugelus_csrsor_iterate         cugelusCcsrsor_iterate
#elif defined(PETSC_USE_REAL_DOUBLE)
#define cugelus_csr_iteration_analysis cugelusZcsrsor_iteration_analysis
#define GELUS_SCALAR_TYPE              gelusDoubleComplex
#define cugelus_csrsor_iterate         cugelusZcsrsor_iterate
#endif
#endif

typedef struct {
  cugelusSorIterationData_t sorIterData;
  gelusSorMethod_t          sorMethod;
  gelusSolveDescription_t   solveDescription;
  gelusSorInitialGuess_t    initialGuess;
  PetscReal                 omega;
  PetscInt                  its;
} PC_SORGELUS;


#undef __FUNCT__
#define __FUNCT__ "PCApply_SORGELUS"
PETSC_EXTERN PetscErrorCode PCApply_SORGELUS(PC pc,Vec x,Vec y)
{
  CUSPARRAY                 *xGPU,*yGPU;
  PetscErrorCode            ierr;
  gelusStatus_t             gStat;
  PC_SORGELUS               *sorGelus  = (PC_SORGELUS*)pc->data;
  cugelusSorIterationData_t h          = sorGelus->sorIterData;
#if defined PETSC_GELUS_USE_LOG
  static int registered = 0;
  static PetscLogEvent cugelusSORIter;
#endif
  PetscFunctionBegin;
  ierr = VecCUSPGetArrayRead(x,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(y,&yGPU);CHKERRQ(ierr);

#if defined PETSC_GELUS_USE_LOG
  if (registered == 0) {
    ierr = PetscLogEventRegister("cugelusSORIter", PC_CLASSID, &cugelusSORIter);CHKERRQ(ierr);
    registered = 1;
  }
  ierr = PetscLogEventBegin(cugelusSORIter,pc,0,0,0);CHKERRQ(ierr);
#endif
  gStat = cugelus_csrsor_iterate((GELUS_SCALAR_TYPE*)yGPU->data().get(),(const GELUS_SCALAR_TYPE*)xGPU->data().get(),sorGelus->initialGuess,sorGelus->its,h);
  if(gStat!=GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusSORIter,pc,0,0,0);CHKERRQ(ierr);
#endif

  ierr = VecCUSPRestoreArrayRead(x,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(y,&yGPU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SORGELUS"
PETSC_EXTERN PetscErrorCode PCSetUp_SORGELUS(PC pc)
{
  gelusStatus_t             gStat            = GELUS_STATUS_SUCCESS;
  cugelusSorIterationData_t sorIterData      = 0;
  gelusSolveDescription_t   solveDescription = 0;
  PC_SORGELUS               *sorGelus        = (PC_SORGELUS*)pc->data;
  PetscErrorCode            ierr             = 0;
  Mat_SeqAIJ                *mat;
  PetscInt                  m,*ai,*aj;
  GELUS_SCALAR_TYPE         *aa;
#if defined PETSC_GELUS_USE_LOG
  static int registered = 0;
  static PetscLogEvent cugelusSORAnalze;
#endif

  PetscFunctionBegin;
  gStat = cugelusCreateSorIterationData(&sorIterData);
  if (gStat != GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Gelus error: %s", gelusGetErrorString(gStat)); }
  sorGelus->sorIterData = sorIterData;

  gStat = gelusCreateSolveDescr(&solveDescription);
  if (gStat != GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Gelus error: %s", gelusGetErrorString(gStat)); }
  sorGelus->solveDescription = solveDescription;

  mat = (Mat_SeqAIJ*)pc->pmat->data;
  m = pc->pmat->rmap->n;
  ai = mat->i;
  aj = mat->j;
  aa = (GELUS_SCALAR_TYPE*)mat->a;
#if defined PETSC_GELUS_USE_LOG
  if (registered == 0) {
    ierr = PetscLogEventRegister("cugelusSORAnalze", PC_CLASSID, &cugelusSORAnalze);CHKERRQ(ierr);
    registered = 1;
  }
  ierr = PetscLogEventBegin(cugelusSORAnalze,pc,0,0,0);CHKERRQ(ierr);
#endif
  gStat = cugelus_csr_iteration_analysis(m,sorGelus->solveDescription,sorGelus->sorMethod,sorGelus->omega,ai,aj,aa,sorGelus->sorIterData);
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusSORAnalze,pc,0,0,0);CHKERRQ(ierr);
#endif
 
  if (gStat != GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Gelus error: %s", gelusGetErrorString(gStat)); }

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_SORGELUS"
PETSC_EXTERN PetscErrorCode PCReset_SORGELUS(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_SORGELUS"
PETSC_EXTERN PetscErrorCode PCDestroy_SORGELUS(PC pc)
{
  PC_SORGELUS               *sorGelus;
  gelusStatus_t             gStat = GELUS_STATUS_SUCCESS;
#if defined PETSC_GELUS_USE_LOG
  PetscErrorCode            ierr = 0;
  static int registered = 0;
  static PetscLogEvent cugelusSORDestr;
#endif
  PetscFunctionBegin;

  sorGelus = (PC_SORGELUS*)pc->data;

  gStat = gelusDestroySolveDescr(sorGelus->solveDescription);
  if (gStat != GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Gelus error: %s", gelusGetErrorString(gStat)); }

#if defined PETSC_GELUS_USE_LOG
  if (registered == 0) {
    ierr = PetscLogEventRegister("cugelusSORDestr", PC_CLASSID, &cugelusSORDestr);CHKERRQ(ierr);
    registered = 1;
  }
  ierr = PetscLogEventBegin(cugelusSORDestr,pc,0,0,0);CHKERRQ(ierr);
#endif
  gStat = cugelusDestroySorIterationData(sorGelus->sorIterData);
  if (gStat != GELUS_STATUS_SUCCESS) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Gelus error: %s", gelusGetErrorString(gStat)); }
#if defined PETSC_GELUS_USE_LOG
  ierr = PetscLogEventEnd(cugelusSORDestr,pc,0,0,0);CHKERRQ(ierr);
#endif
 
  PetscFree(sorGelus);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_SORGELUS"
PETSC_EXTERN PetscErrorCode PCSetFromOptions_SORGELUS(PC pc)
{
  PC_SORGELUS               *sorGelus        = (PC_SORGELUS*)pc->data;
  PetscErrorCode            ierr             = 0;
  const char                *mansection      = "No section in manual";
  PetscBool                 flg, init_guess_non_zero;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("GELUS (S)SOR options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_sorgelus_omega","relaxation factor (0 < omega < 2)",mansection,sorGelus->omega,&sorGelus->omega,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_sorgelus_its","number of inner SOR iterations",mansection,sorGelus->its,&sorGelus->its,0);CHKERRQ(ierr);

  if(sorGelus->initialGuess == GELUS_SOR_INITIAL_GUESS_NONZERO)
    init_guess_non_zero = PETSC_TRUE;
  else
    init_guess_non_zero = PETSC_FALSE;

  ierr = PetscOptionsBool("-pc_sorgelus_initial_guess_nonzero","Use the contents of the solution vector for initial guess",mansection,init_guess_non_zero,&init_guess_non_zero,0);CHKERRQ(ierr);
  if(init_guess_non_zero)
    sorGelus->initialGuess = GELUS_SOR_INITIAL_GUESS_NONZERO;
  else
    sorGelus->initialGuess = GELUS_SOR_INITIAL_GUESS_ZERO;

  ierr = PetscOptionsBoolGroupBegin("-pc_sorgelus_symmetric","SSOR, not SOR",mansection,&flg);CHKERRQ(ierr);
  if (flg) sorGelus->sorMethod = GELUS_SOR_SYMMETRIC;
  ierr = PetscOptionsBoolGroup("-pc_sorgelus_backward","use backward sweep instead of forward",mansection,&flg);CHKERRQ(ierr);
  if (flg) sorGelus->sorMethod = GELUS_SOR_BACKWARD;
  ierr = PetscOptionsBoolGroup("-pc_sorgelus_forward","use forward sweep",mansection,&flg);CHKERRQ(ierr);
  if (flg) sorGelus->sorMethod = GELUS_SOR_FORWARD;
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCreate_SORGELUS"
PETSC_EXTERN PetscErrorCode PCCreate_SORGELUS(PC pc)
{
  PC_SORGELUS               *sorGelus;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&sorGelus);CHKERRQ(ierr);
  pc->data = sorGelus;

  sorGelus->sorIterData      = 0;
  sorGelus->sorMethod        = GELUS_SOR_SYMMETRIC;
  sorGelus->solveDescription = 0;
  sorGelus->initialGuess     = GELUS_SOR_INITIAL_GUESS_NONZERO;
  sorGelus->omega            = 1;
  sorGelus->its              = 1;

  pc->ops->apply               = PCApply_SORGELUS;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_SORGELUS;
  pc->ops->reset               = PCReset_SORGELUS;
  pc->ops->destroy             = PCDestroy_SORGELUS;
  pc->ops->setfromoptions      = PCSetFromOptions_SORGELUS;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(ierr);
}

