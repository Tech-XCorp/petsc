PETSc with Tech-X's GPU improvements
------------------------------------

This repository makes available several patches for GPU functionality in
the [PETSc library](http://www.mcs.anl.gov/petsc/).  The main PETSc repository
is at [https://bitbucket.org/petsc/petsc](https://bitbucket.org/petsc/petsc).

The branch gpu-master was created off a fairly recent version of PETSc's
master branch.  Our GPU related patches have been replayed onto this
branch and an attempt has been made to resolve conflicts in a meaningful
way.  Note that there is no guarantee whatsoever that conflicts will be
resolved in an equivalent way when this code gets merge into
petsc/master.

Important information:
----------------------

- The patches contained in this branch are for early adopters.  They
  have not yet undergone rigorous review and testing.  
- *Never* create a branch off of gpu-master with the intent of creating
  a patch for PETSc.  The gpu-master branch contains experimental code
  which may not be suitable for inclusion in petsc/master.  It will be
  frequently rebased to keep up with changes in petsc/master.  It should
  be treated similarly to petsc/next!

