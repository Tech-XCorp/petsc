/*
      This example tests the signal handler.
*/
#include "petsc.h"
#include <stdio.h>

int CreateError(int n)
{
  int    ierr;
  double *x = 0;
  if (!n) x[0] = 100.; 
  ierr = CreateError(n-1); CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,0,0);
  fprintf(stdout,"Demonstrates how PETSc may trap error interupts\n");
  fprintf(stdout,"The error below is contrived to test the code!\n");
  fflush(stdout);
  ierr = CreateError(5); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
