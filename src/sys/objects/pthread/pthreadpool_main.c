
#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

static int*         pVal_main;

#define CACHE_LINE_SIZE 64  /* used by 'chain', 'main','tree' thread pools */

typedef void* (*pfunc)(void*);

/* main thread pool data structure */
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  pfunc *funcArr;
  void** pdata;
  PetscBool** arrThreadReady;
} sjob_main;
sjob_main job_main;

static char* arrmutex;
static char* arrcond1;
static char* arrcond2;
static char* arrstart;
static char* arrready;

/* 
   ----------------------------
   'Main' Thread Pool Functions
   ---------------------------- 
*/
void* PetscThreadFunc_Main(void* arg) {

  int* pId = (int*)arg;
  int ThreadId = *pId;

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  DoCoreAffinity();
#endif

  pthread_mutex_lock(job_main.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_main.arrThreadReady[ThreadId]) = PETSC_TRUE;
  /* tell the BOSS that you're ready to work before you go to sleep */
  pthread_cond_signal(job_main.cond1array[ThreadId]);

  /* the while loop needs to have an exit
     the 'main' thread can terminate all the threads by performing a broadcast
     and calling FuncFinish */
  while(PetscThreadGo) {
    /* need to check the condition to ensure we don't have to wait
       waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_main.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /* upon entry, atomically releases the lock and blocks
       upon return, has the lock */
        pthread_cond_wait(job_main.cond2array[ThreadId],job_main.mutexarray[ThreadId]);
	/* (job_main.arrThreadReady[ThreadId])   = PETSC_FALSE; */
    }
    pthread_mutex_unlock(job_main.mutexarray[ThreadId]);
    if(job_main.funcArr[ThreadId+PetscMainThreadShareWork]) {
      job_main.funcArr[ThreadId+PetscMainThreadShareWork](job_main.pdata[ThreadId+PetscMainThreadShareWork]);
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      pthread_mutex_lock(job_main.mutexarray[ThreadId]);
      *(job_main.arrThreadReady[ThreadId]) = PETSC_TRUE;
      /* tell the BOSS that you're ready to work before you go to sleep */
      pthread_cond_signal(job_main.cond1array[ThreadId]);
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_Main"
PetscErrorCode PetscThreadsSynchronizationInitialize_Main(PetscInt N) 
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MEMALIGN)
  size_t Val1 = (size_t)CACHE_LINE_SIZE;
  size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
  arrmutex = (char*)memalign(Val1,Val2);
  arrcond1 = (char*)memalign(Val1,Val2);
  arrcond2 = (char*)memalign(Val1,Val2);
  arrstart = (char*)memalign(Val1,Val2);
  arrready = (char*)memalign(Val1,Val2);
#else
  size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
  arrmutex = (char*)malloc(Val2);
  arrcond1 = (char*)malloc(Val2);
  arrcond2 = (char*)malloc(Val2);
  arrstart = (char*)malloc(Val2);
  arrready = (char*)malloc(Val2);
#endif
  
  job_main.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
  job_main.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_main.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_main.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
  /* initialize job structure */
  for(i=0; i<PetscMaxThreads; i++) {
    job_main.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
    job_main.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
    job_main.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
    job_main.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
  }
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_mutex_init(job_main.mutexarray[i],NULL);CHKERRQ(ierr);
    ierr = pthread_cond_init(job_main.cond1array[i],NULL);CHKERRQ(ierr);
    ierr = pthread_cond_init(job_main.cond2array[i],NULL);CHKERRQ(ierr);
    *(job_main.arrThreadReady[i])    = PETSC_FALSE;
  }
  job_main.funcArr = (pfunc*)malloc((N+PetscMainThreadShareWork)*sizeof(pfunc));
  job_main.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  pVal_main = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  /* create threads */
  for(i=0; i<N; i++) {
    pVal_main[i] = i;
    job_main.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_main.pdata[i+PetscMainThreadShareWork] = NULL;
    ierr = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_main[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationFinalize_Main"
PetscErrorCode PetscThreadsSynchronizationFinalize_Main() {
  int            i;
  void*          jstatus;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscThreadsRunKernel(FuncFinish,NULL,PetscMaxThreads,PETSC_NULL);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);CHKERRQ(ierr);
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_main.funcArr);
  free(job_main.pdata);
  free(pVal_main);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsWait_Main"
void* PetscThreadsWait_Main(void* arg) {

  PetscInt i;
  for(i=0; i<PetscMaxThreads; i++) {
    pthread_mutex_lock(job_main.mutexarray[i]);
    while(*(job_main.arrThreadReady[i])==PETSC_FALSE) {
      pthread_cond_wait(job_main.cond1array[i],job_main.mutexarray[i]);
    }
    pthread_mutex_unlock(job_main.mutexarray[i]);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_Main"
PetscErrorCode PetscThreadsRunKernel_Main(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) {
  int i,j,issetaffinity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadsWait(NULL); /* you know everyone is waiting to be signalled! */
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_main.arrThreadReady[i]) = PETSC_FALSE; /* why do this?  suppose you get into PetscThreadsWait first */
  }
  /* tell the threads to go to work */
  for(i=0; i<PetscMaxThreads; i++) {
    if(pFunc == FuncFinish) {
      job_main.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_main.pdata[i+PetscMainThreadShareWork] = NULL;
    } else {
      issetaffinity=0;
      for(j=PetscMainThreadShareWork; j < n; j++) {
	if(cpu_affinity[j] == ThreadCoreAffinity[i]) {
	  job_main.funcArr[i+PetscMainThreadShareWork] = pFunc;
	  job_main.pdata[i+PetscMainThreadShareWork] = data[j];
	  issetaffinity=1;
	}
      }
      if(!issetaffinity) {
	job_main.funcArr[i+PetscMainThreadShareWork] = NULL;
	job_main.pdata[i+PetscMainThreadShareWork] = NULL;
      }
    }

    ierr = pthread_cond_signal(job_main.cond2array[i]);CHKERRQ(ierr);
  }
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      job_main.funcArr[0] = pFunc;
      job_main.pdata[0] = data[0];
      job_main.funcArr[0](job_main.pdata[0]);
    }
    PetscThreadsWait(NULL); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  PetscFunctionReturn(0);
}