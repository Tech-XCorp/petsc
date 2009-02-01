#include "private/fortranimpl.h"
#include "petscmesh.h"
#include "../src/sys/f90/f90impl.h"

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" { 
#endif 
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
} 
#endif 

#else

#define PetscToPointer(a) (*(long *)(a))
#define PetscFromPointer(a) (long)(a)
#define PetscRmPointer(a)
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define meshgetcoordinatesf90_     MESHGETCOORDINATESF90
#define meshrestorecoordinatesf90_ MESHRESTORECOORDINATESF90
#define meshgetelementsf90_        MESHGETELEMENTSF90
#define meshrestoreelementsf90_    MESHRESTOREELEMENTSF90
#define meshrestrictclosure_       MESHRESTRICTCLOSURE
#define meshrestrictclosureint_    MESHRESTRICTCLOSUREINT
#define meshupdateclosure_         MESHUPDATECLOSURE
#define meshupdateclosureint_      MESHUPDATECLOSUREINT
#define meshupdateaddclosure_      MESHUPDATEADDCLOSURE
#define sectiongetarrayf90_        SECTIONGETARRAYF90
#define sectiongetarray1df90_      SECTIONGETARRAY1DF90
#define bcsectiongetarrayf90_      BCSECTIONGETARRAYF90
#define bcsectiongetarray1df90_    BCSECTIONGETARRAY1DF90
#define bcsectionrealgetarrayf90_  BCSECTIONREALGETARRAYF90
#define bcfuncgetarrayf90_         BCFUNCGETARRAYF90
#define meshgetlabelids_           MESHGETLABELIDS
#define meshgetstratum_            MESHGETSTRATUM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define meshgetcoordinatesf90_     meshgetcoordinatesf90
#define meshrestorecoordinatesf90_ meshrestorecoordinatesf90
#define meshgetelementsf90_        meshgetelementsf90
#define meshrestoreelementsf90_    meshrestoreelementsf90
#define meshrestrictclosure_       meshrestrictclosure
#define meshrestrictclosureint_    meshrestrictclosureint
#define meshupdateclosure_         meshupdateclosure
#define meshupdateclosureint_      meshupdateclosureint
#define meshupdateaddclosure_      meshupdateaddclosure
#define sectiongetarrayf90_        sectiongetarrayf90
#define sectiongetarray1df90_      sectiongetarray1df90
#define bcsectiongetarrayf90_      bcsectiongetarrayf90
#define bcsectiongetarray1df90_    bcsectiongetarray1df90
#define bcsectionrealgetarrayf90_  bcsectionrealgetarrayf90
#define bcfuncgetarrayf90_         bcfuncgetarrayf90
#define meshgetlabelids_           meshgetlabelids
#define meshgetstratum_            meshgetstratum
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL meshgetcoordinatesf90_(Mesh *mesh,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  PetscInt   n, d;
  *__ierr = MeshGetCoordinates(*mesh,PETSC_TRUE,&n,&d,&c); if (*__ierr) return;
  *__ierr = F90Array2dCreate(c,PETSC_REAL,1,n,1,d,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshrestorecoordinatesf90_(Mesh *x,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  *__ierr = F90Array2dAccess(ptr,PETSC_REAL,(void**)&c PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr,PETSC_REAL PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(c);
}
void PETSC_STDCALL meshgetelementsf90_(Mesh *mesh,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v;
  PetscInt   n, c;
  *__ierr = MeshGetElements(*mesh,PETSC_TRUE,&n,&c,&v); if (*__ierr) return;
  *__ierr = F90Array2dCreate(v,PETSC_INT,1,n,1,c,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshrestoreelementsf90_(Mesh *x,F90Array2d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *v;
  *__ierr = F90Array2dAccess(ptr,PETSC_INT,(void**)&v PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(v);
}

void PETSC_STDCALL meshrestrictclosure_(Mesh mesh, SectionReal section, int *point,int *size,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *c;

  // Should be able to get array size
  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshRestrictClosure((Mesh) PetscToPointer(mesh), (SectionReal) PetscToPointer(section),*point,*size,c); if (*ierr) return;
  // *ierr = F90Array1dCreate(const_cast<PetscScalar *>(c),PETSC_SCALAR,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshrestrictclosureint_(Mesh mesh, SectionInt section, int *point,int *size,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *c;

  // Should be able to get array size
  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshRestrictClosureInt((Mesh) PetscToPointer(mesh), (SectionInt) PetscToPointer(section),*point,*size,c); if (*ierr) return;
  // *ierr = F90Array1dCreate(const_cast<PetscScalar *>(c),PETSC_SCALAR,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshupdateclosure_(Mesh mesh, SectionReal section, int *point,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *c;

  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshUpdateClosure((Mesh) PetscToPointer(mesh), (SectionReal) PetscToPointer(section),*point,c); if (*ierr) return;
}
void PETSC_STDCALL meshupdateclosureint_(Mesh mesh, SectionInt section, int *point,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *c;

  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshUpdateClosureInt((Mesh) PetscToPointer(mesh), (SectionInt) PetscToPointer(section),*point,c); if (*ierr) return;
}
void PETSC_STDCALL meshupdateaddclosure_(Mesh mesh, SectionReal section, int *point,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *c;

  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &c PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshUpdateAddClosure((Mesh) PetscToPointer(mesh), (SectionReal) PetscToPointer(section),*point,c); if (*ierr) return;
}
#if 0
void PETSC_STDCALL meshrestoreclosuref90_(Mesh mesh,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *c;
  *__ierr = F90Array1dAccess(ptr,PETSC_REAL,(void**)&c PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_REAL PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(c);
}
#endif
void PETSC_STDCALL sectiongetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *a;
  PetscInt     n, d;
  char        *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL sectiongetarray1df90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array1d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *a;
  PetscInt     n, d;
  char        *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array1dCreate(a,PETSC_SCALAR,1,n*d,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectiongetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *a;
  PetscInt  n, d;
  char     *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_INT,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectiongetarray1df90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array1d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *a;
  PetscInt  n, d;
  char     *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array1dCreate(a,PETSC_INT,1,n*d,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcsectionrealgetarrayf90_(Mesh *mesh,CHAR name PETSC_MIXED_LEN(len),F90Array2d *ptr,int *ierr PETSC_END_LEN(len) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *a;
  PetscInt   n, d;
  char      *nF;
  FIXCHAR(name,len,nF);
  *ierr = BCSectionRealGetArray(*mesh,nF,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name,nF);
}
void PETSC_STDCALL bcfuncgetarrayf90_(Mesh *mesh,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *a;
  PetscInt     n, d;
  *ierr = BCFUNCGetArray(*mesh,&n,&d,&a); if (*ierr) return;
  *ierr = F90Array2dCreate(a,PETSC_SCALAR,1,d,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL meshgetlabelids_(Mesh *mesh, CHAR name PETSC_MIXED_LEN(lenN), F90Array1d *ptr, int *ierr PETSC_END_LEN(lenN) PETSC_F90_2PTR_PROTO(ptrd)){
  char     *pN;
  PetscInt *ids;
  FIXCHAR(name,lenN,pN);
  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &ids PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshGetLabelIds(*mesh,pN, ids);
  FREECHAR(name,pN);
}
void PETSC_STDCALL meshgetstratum_(Mesh *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, F90Array1d *ptr, int *ierr PETSC_END_LEN(lenN) PETSC_F90_2PTR_PROTO(ptrd)){
  char     *pN;
  PetscInt *points;
  FIXCHAR(name,lenN,pN);
  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MeshGetStratum(*mesh,pN, *value, points);
  FREECHAR(name,pN);
}

EXTERN_C_END
