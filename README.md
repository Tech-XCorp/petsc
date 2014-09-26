PETSc with Tech-X's GPU improvements
------------------------------------

This repository makes available several patches for GPU functionality in
the [PETSc library](http://www.mcs.anl.gov/petsc/).  The main PETSc repository
is at [https://bitbucket.org/petsc/petsc](https://bitbucket.org/petsc/petsc).

The branch gpu-master contains several branches that are currently under
review by the PETSc team.  gpu-master was created off a fairly recent
version of PETSc's master branch.  

Important information:
----------------------

- The patches contained in this branch are for *early adopters* and
  developers.  They have not yet undergone rigorous review and testing.  
  You should know your way around PETSc and you should be familiar with
  the PETSc development process.
- *Never* create a branch off of gpu-master with the intent of creating
  a patch for PETSc.  Such branches should *always* be made off of
  petsc/master.  The gpu-master branch contains experimental code
  which may not be suitable for inclusion in petsc/master.  It will be
  frequently rebased to keep up with changes in petsc/master.  It should
  be treated similarly to petsc/next!

