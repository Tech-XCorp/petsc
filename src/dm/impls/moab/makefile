#requirespackage  'PETSC_HAVE_MOAB'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECXX= dmmoab.cxx dmmbvec.cxx dmmbmat.cxx dmmbfield.cxx dmmbio.cxx dmmbutil.cxx
SOURCEF  =
SOURCEH  =
DIRS     = examples
LIBBASE  = libpetscdm
MANSEC   = DM
LOCDIR   = src/dm/impls/moab/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

