#!/usr/bin/env python

"""
PETSc: Portable, Extensible Toolkit for Scientific Computation
==============================================================

The Portable, Extensible Toolkit for Scientific Computation (PETSc),
is a suite of data structures and routines for the scalable (parallel)
solution of scientific applications modeled by partial differential
equations. It employs the Message Passing Interface (MPI) standard for
all message-passing communication.
"""

import sys, os
from distutils.core import setup
from distutils.util import get_platform
from distutils.spawn import find_executable
from distutils.command.build import build as _build
if 'setuptools' in sys.modules:
    from setuptools.command.install import install as _install
else:
    from distutils.command.install import install as _install
from distutils.command.sdist import sdist as _sdist
from distutils import log

PETSC_DIR  = None
PETSC_ARCH = None

init_py = """\
# Author:  PETSc Team
# Contact: petsc-users@mcs.anl.gov

def get_petsc_dir():
    import os
    return os.path.dirname(__file__)

def get_petsc_arch():
    return ''

def get_config():
    conf = {}
    conf['petsc_dir'] = get_petsc_dir()
    return conf
"""

metadata = {
    'provides' : ['petsc'],
    'requires' : [],
}

def bootstrap():
    # Set PETSC_DIR and PETSC_ARCH, 
    global PETSC_DIR, PETSC_ARCH
    PETSC_DIR  = os.path.abspath(os.getcwd())
    PETSC_ARCH = get_platform() + '-python'
    os.environ['PETSC_DIR']  = PETSC_DIR
    os.environ['PETSC_ARCH'] = PETSC_ARCH
    sys.path.insert(0, os.path.join(PETSC_DIR, 'config'))
    # Generate package __init__.py file
    try:
        if not os.path.exists(PETSC_ARCH):
            os.mkdir(PETSC_ARCH)
        pkgfile = os.path.join(PETSC_ARCH, '__init__.py')
        if not os.path.exists(pkgfile):
            open(pkgfile, 'wt').write(init_py)
    except:
        pass
    # Simple-minded lookup for MPI and mpi4py
    mpi4py = mpicc = None
    try:
        import mpi4py
        conf = mpi4py.get_config()
        mpicc = conf.get('mpicc')
    except ImportError: # mpi4py is not installed
        mpicc = os.environ.get('MPICC') or find_executable('mpicc')
    except AttributeError: # mpi4py is too old
        pass
    if not mpi4py and mpicc:
        if (('distribute' in sys.modules) or
            ('setuptools' in sys.modules)):
            metadata['install_requires']= ['mpi4py>=1.2.2']

def config(dry_run=False):
    log.info('PETSc: configure')
    if dry_run: return
    options = [
        'PETSC_ARCH='+os.environ['PETSC_ARCH'],
        '--with-shared-libraries',
        '--with-cmake=0', # not needed
        ]
    # MPI
    try:
        import mpi4py
        conf = mpi4py.get_config()
        mpicc = conf.get('mpicc')
    except (ImportError, AttributeError):
        mpicc = os.environ.get('MPICC') or find_executable('mpicc')
    if mpicc:
        options.append('--with-cc='+mpicc)
    else:
        options.append('--with-mpi=0')
    options.extend([
        '--with-fc=0',    # XXX mpif90?
        '--with-cxx=0',   # XXX mpicxx?
        ])
    # Run PETSc configure
    import configure
    configure.petsc_configure(options)
    import logger
    logger.Logger.defaultLog = None

def build(dry_run=False):
    log.info('PETSc: build')
    if dry_run: return
    # Run PETSc builder
    import builder
    builder.PETScMaker().run()
    import logger
    logger.Logger.defaultLog = None

def install(dest_dir, prefix=None, dry_run=False):
    log.info('PETSc: install')
    if dry_run: return
    if prefix is None:
        prefix = dest_dir
    options = [
        '--destDir=' + dest_dir,
        '--prefix='  + prefix
        ]
    # Run PETSc installer
    import install
    install.Installer(options).run()
    import logger
    logger.Logger.defaultLog = None
    # temporary hack - delete log files created by BuildSystem
    delfiles=['RDict.db','RDict.log',
              'build.log','default.log',
              'build.log.bkp','default.log.bkp']
    for delfile in delfiles:
        try:
            if (os.path.exists(delfile) and
                os.stat(delfile).st_uid==0):
                os.remove(delfile)
        except:
            pass

class cmd_build(_build):

    def finalize_options(self):
        if self.build_base is None:
            self.build_base= 'build'
        self.build_base = os.path.join(
            os.environ['PETSC_ARCH'], self.build_base)
        _build.finalize_options(self)

    def run(self):
        _build.run(self)
        wdir = os.getcwd()
        pdir = os.environ['PETSC_DIR']
        try:
            os.chdir(pdir)
            config(self.dry_run)
            build(self.dry_run)
        finally:
            os.chdir(wdir)

class cmd_install(_install):

    def run(self):
        _install.run(self)
        root_dir = self.install_platlib
        dest_dir = os.path.join(root_dir, 'petsc')
        bdist_base = self.get_finalized_command('bdist').bdist_base
        if dest_dir.startswith(bdist_base):
            prefix = dest_dir[len(bdist_base)+1:]
            prefix = prefix[prefix.index(os.path.sep):]
        else:
            prefix = dest_dir
        dest_dir = os.path.abspath(dest_dir)
        prefix = os.path.abspath(prefix)
        wdir = os.getcwd()
        pdir = os.environ['PETSC_DIR']
        try:
            os.chdir(pdir)
            install(dest_dir, prefix, self.dry_run)
        finally:
            os.chdir(wdir)

class cmd_sdist(_sdist):

    def initialize_options(self):
        _sdist.initialize_options(self)
        self.force_manifest = 1
        self.template = os.path.join('config', 'MANIFEST.in')

    def run(self):
        _sdist.run(self)
        try:
            os.remove(self.manifest)
        except Exception:
            pass

def version():
    return '3.2.dev1' # XXX should parse include/petscversion.h

def tarball():
    return None
    return ('http://ftp.mcs.anl.gov/pub/petsc/<XXX>/' # XXX fix this line
            'petsc-lite-%s.tar.gz' % version() )

description = __doc__.split('\n')[1:-1]; del description[1:3]
classifiers = """
License :: Public Domain
Operating System :: POSIX
Intended Audience :: Developers
Intended Audience :: Science/Research
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

bootstrap()
setup(name='petsc',
      version=version(),
      description=description.pop(0),
      long_description='\n'.join(description),
      classifiers= classifiers.split('\n')[1:-1],
      keywords = ['PETSc', 'MPI'],
      platforms=['POSIX'],
      license='PETSc',

      url='http://www.mcs.anl.gov/petsc/',
      download_url=tarball(),

      author='PETSc Team',
      author_email='petsc-users@mcs.anl.gov',
      maintainer='Lisandro Dalcin',
      maintainer_email='dalcinl@gmail.com',

      packages = ['petsc'],
      package_dir = {'petsc': os.environ['PETSC_ARCH']},
      cmdclass={
        'build': cmd_build,
        'install': cmd_install,
        'sdist': cmd_sdist,
        },
      **metadata)