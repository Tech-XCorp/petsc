import config.package
import config.base
import os.path

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes         = ['cuGelus.h']
    self.includedir       = ['', 'include']
    self.liblist          = [['libcugelus','libgelusbase.a']]
    self.libdir           = 'lib'
    self.functions        = ['cugelusCreateIluSolveData']
    self.cxx              = 1
    self.archIndependent  = 1
    self.worksonWindows   = 1
    self.requires32bitint = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('PETSc.packages.cuda', self)
    self.openmp = framework.require('PETSc.packages.openmp', self)
    self.deps = [self.cuda, self.openmp]
    return

