#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.31 1996/07/08 01:36:09 curfman Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "petsc.h"
#include "../matimpl.h"
#include "sys.h"
#include "pinclude/pviewer.h"

extern int MatLoad_MPIRowbs(Viewer,MatType,Mat*);
extern int MatLoad_SeqAIJ(Viewer,MatType,Mat*);
extern int MatLoad_MPIAIJ(Viewer,MatType,Mat*);
extern int MatLoad_SeqBDiag(Viewer,MatType,Mat*);
extern int MatLoad_MPIBDiag(Viewer,MatType,Mat*);
extern int MatLoad_SeqDense(Viewer,MatType,Mat*);
extern int MatLoad_MPIDense(Viewer,MatType,Mat*);
extern int MatLoad_SeqBAIJ(Viewer,MatType,Mat*);
extern int MatLoad_MPIBAIJ(Viewer,MatType,Mat*);

extern int MatLoadGetInfo_Private(Viewer);

/*@C
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().  The matrix format is determined from the options database.
   Generates a parallel MPI matrix if the communicator has more than one
   processor.  The default matrix type is AIJ.

   Input Parameters:
.  viewer - binary file viewer, created with ViewerFileOpenBinary()
.  outtype - type of matrix desired, for example MATSEQAIJ,
   MATMPIROWBS, etc.  See types in petsc/include/mat.h.

   Output Parameters:
.  newmat - new matrix

   Basic Options Database Keys:
   These options use MatCreateSeqXXX or MatCreateMPIXXX,
   depending on the communicator, comm.
$    -mat_aij      : AIJ type
$    -mat_baij     : block AIJ type
$    -mat_dense    : dense type
$    -mat_bdiag    : block diagonal type

   More Options Database Keys:
$    -mat_seqaij   : AIJ type
$    -mat_mpiaij   : parallel AIJ type
$    -mat_seqbaij  : block AIJ type
$    -mat_mpibaij  : parallel block AIJ type
$    -mat_seqbdiag : block diagonal type
$    -mat_mpibdiag : parallel block diagonal type
$    -mat_mpirowbs : parallel rowbs type
$    -mat_seqdense : dense type
$    -mat_mpidense : parallel dense type

   More Options Database Keys:
   Used with block matrix formats (MATSEQBAIJ, MATMPIBDIAG, ...) to specify
   block size
$    -matload_block_size <bs>

   Used to specify block diagonal numbers for MATSEQBDIAG and MATMPIBDIAG formats
$    -matload_bdiag_diags <s1,s2,s3,...>

   Notes:
   In parallel, each processor can load a subset of rows (or the
   entire matrix).  This routine is especially useful when a large
   matrix is stored on disk and only part of it is desired on each
   processor.  For example, a parallel solver may access only some of
   the rows from each processor.  The algorithm used here reads
   relatively small blocks of data rather than reading the entire
   matrix and then subsetting it.

   Notes for advanced users:
   Most users should not need to know the details of the binary storage
   format, since MatLoad() and MatView() completely hide these details.
   But for anyone who's interested, the standard binary matrix storage
   format is

$    int    MAT_COOKIE
$    int    number of rows
$    int    number of columns
$    int    total number of nonzeros
$    int    *number nonzeros in each row
$    int    *column indices of all nonzeros (starting index is zero)
$    Scalar *values of all nonzeros

.keywords: matrix, load, binary, input

.seealso: ViewerFileOpenBinary(), MatView(), VecLoad() 
 @*/  
int MatLoad(Viewer viewer,MatType outtype,Mat *newmat)
{
  int         ierr,set;
  MatType     type;
  ViewerType  vtype;
  MPI_Comm    comm;
  *newmat = 0;

  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary()");

  PetscObjectGetComm((PetscObject)viewer,&comm);
  ierr = MatGetTypeFromOptions(comm,0,&type,&set); CHKERRQ(ierr);
  if (!set) type = outtype;

  ierr = MatLoadGetInfo_Private(viewer); CHKERRQ(ierr);

  PLogEventBegin(MAT_Load,viewer,0,0,0);

  if (type == MATSEQAIJ) {
    ierr = MatLoad_SeqAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIAIJ) {
    ierr = MatLoad_MPIAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQBDIAG) {
    ierr = MatLoad_SeqBDiag(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIBDIAG) {
    ierr = MatLoad_MPIBDiag(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQDENSE) {
    ierr = MatLoad_SeqDense(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIDENSE) {
    ierr = MatLoad_MPIDense(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIROWBS) {
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
    ierr = MatLoad_MPIRowbs(viewer,type,newmat); CHKERRQ(ierr);
#else
    SETERRQ(1,"MatLoad:MATMPIROWBS does not support complex numbers");
#endif
  }
  else if (type == MATSEQBAIJ) {
    ierr = MatLoad_SeqBAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIBAIJ) {
    ierr = MatLoad_MPIBAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else {
    SETERRQ(1,"MatLoad: cannot load with that matrix type yet");
  }

  PLogEventEnd(MAT_Load,viewer,0,0,0);
  return 0;
}

/*
    MatLoadGetInfo_Private - Loads the matrix options from the name.info file
  if it exists.

*/
int MatLoadGetInfo_Private(Viewer viewer)
{
  FILE *file;
  char string[128],*first,*second,*final;
  int  len,ierr,flg;

  ierr = OptionsHasName(PETSC_NULL,"-matload_ignore_info",&flg);CHKERRQ(ierr);
  if (flg) return 0;

  ierr = ViewerBinaryGetInfoPointer(viewer,&file); CHKERRQ(ierr);
  if (!file) return 0;

  /* read rows of the file adding them to options database */
  while (fgets(string,128,file)) {
    /* Comments are indicated by #, ! or % in the first column */
    if (string[0] == '#') continue;
    if (string[0] == '!') continue;
    if (string[0] == '%') continue;
    first = PetscStrtok(string," ");
    second = PetscStrtok(0," ");
    if (first && first[0] == '-') {
      if (second) {final = second;} else {final = first;}
      len = PetscStrlen(final);
      while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
        len--; final[len] = 0;
      }
      ierr = OptionsSetValue(first,second); CHKERRQ(ierr);
    }
  }
  return 0;

}
