#ifndef lint
static char vcid[] = "$Id: bjacobi.c,v 1.28 1995/07/20 03:58:50 bsmith Exp bsmith $";
#endif
/*
   Defines a block Jacobi preconditioner.

   This is far from object oriented. We include the matrix 
  details in the code. This is because otherwise we would have to 
  include knowledge of SLES in the matrix code. In other words,
  it is a loop of objects, not a tree, so inheritence is a bad joke.
*/
#include "src/mat/matimpl.h"
#include "pcimpl.h"
#include "bjacobi.h"
#include "pviewer.h"

extern int PCSetUp_BJacobiMPIAIJ(PC);
extern int PCSetUp_BJacobiMPIRow(PC);
extern int PCSetUp_BJacobiMPIBDiag(PC);

static int PCSetUp_BJacobi(PC pc)
{
  Mat        mat = pc->mat;
  if (mat->type == MATMPIAIJ)        return PCSetUp_BJacobiMPIAIJ(pc);
  else if (mat->type == MATMPIROW)   return PCSetUp_BJacobiMPIRow(pc);
  else if (mat->type == MATMPIBDIAG) return PCSetUp_BJacobiMPIBDiag(pc);
  SETERRQ(1,"PCSetUp_BJacobi: Cannot use block Jacobi on this matrix type\n");
}

/* Default destroy, if it has never been setup */
static int PCDestroy_BJacobi(PetscObject obj)
{
  PC pc = (PC) obj;
  PC_BJacobi *jac = (PC_BJacobi *) pc->data;
  PETSCFREE(jac);
  return 0;
}

static int PCSetFromOptions_BJacobi(PC pc)
{
  int        blocks;

  if (OptionsGetInt(pc->prefix,"-pc_bjacobi_blocks",&blocks)) {
    PCBJacobiSetBlocks(pc,blocks);
  }
  if (OptionsHasName(pc->prefix,"-pc_bjacobi_truelocal")) {
    PCBJacobiSetUseTrueLocal(pc);
  }
  return 0;
}

/*@
   PCBJacobiSetUseTrueLocal - Sets a flag to indicate that the block 
   problem is associated with the linear system matrix instead of the
   default (where it is associated with the preconditioning matrix).

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_bjacobi_truelocal

   Note:
   For the common case in which the preconditioning and linear 
   system matrices are identical, this routine is unnecessary.

.keywords:  block, Jacobi, set, true, local, flag

.seealso: PCSetOperators(), PCBJacobiSetBlocks()
@*/
int PCBJacobiSetUseTrueLocal(PC pc)
{
  PC_BJacobi   *jac;
  VALIDHEADER(pc,PC_COOKIE);
  if (pc->type != PCBJACOBI) return 0;
  jac = (PC_BJacobi *) pc->data;
  jac->usetruelocal = 1;
  return 0;
}
  
static int PCPrintHelp_BJacobi(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  MPIU_printf(pc->comm," %spc_bjacobi_blocks blks: blocks in preconditioner\n",p);
  MPIU_printf(pc->comm, " %spc_bjacobi_truelocal: use blocks from the local linear\
 system matrix \n      instead of the preconditioning matrix\n",p);
  MPIU_printf(pc->comm," %ssub : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (i.e., -sub_ksp_method\
 <meth>)\n",p);
  return 0;
}


int PCCreate_BJacobi(PC pc)
{
  PC_BJacobi   *jac = PETSCNEW(PC_BJacobi); CHKPTRQ(jac);
  pc->apply         = 0;
  pc->setup         = PCSetUp_BJacobi;
  pc->destroy       = PCDestroy_BJacobi;
  pc->setfrom       = PCSetFromOptions_BJacobi;
  pc->printhelp     = PCPrintHelp_BJacobi;
  pc->type          = PCBJACOBI;
  pc->data          = (void *) jac;
  pc->view          = 0;
  jac->n            = 0;
  jac->n_local      = 1;
  jac->sles         = 0;
  jac->usetruelocal = 0;
  return 0;
}
/*@
   PCBJacobiSetBlocks - Sets the number of blocks for the block Jacobi
   preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  blocks - the number of blocks

   Options Database Key:
$  -pc_bjacobi_blocks  blocks

.keywords:  set, number, Jacobi, blocks

.seealso: PCBJacobiSetUseTrueLocal()
@*/
int PCBJacobiSetBlocks(PC pc, int blocks)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  jac->n = blocks;
  return 0;
}
