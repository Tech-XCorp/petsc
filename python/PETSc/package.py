import config.base
import os
import re
import sys

class Package(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.clanguage    = self.framework.require('PETSc.utilities.clanguage',self)    
    self.functions    = self.framework.require('config.functions',         self)
    self.found        = 0
    self.lib          = []
    # this packages libraries and all those it depends on
    self.dlib         = []
    self.include      = []
    self.name         = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    # these are optional items set in the particular packages file
    self.complex      = 0
    self.download     = []
    
  def __str__(self):
    '''Prints the location of the packages includes and libraries'''
    output=''
    if self.found:
      output  = self.name+':\n'
      if self.include: output += '  Includes: '+str(self.include)+'\n'
      if self.lib:     output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def setupHelp(self,help):
    '''Prints help messages for the package'''
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,0,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    if self.download:
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Download and install '+self.name))
    return

  def checkInclude(self,incl,hfile,otherIncludes = None):
    '''Checks if a particular include file can be found along particular include paths'''
    if hasattr(self,'mpi'):    incl.extend(self.mpi.include)
    if otherIncludes:          incl.extend(otherIncludes)
    oldFlags = self.framework.argDB['CPPFLAGS']
    self.framework.argDB['CPPFLAGS'] += ' '.join([self.libraries.getIncludeArgument(inc) for inc in incl])
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+str(incl)+'\n')
    return found

  def downLoad(self):
    '''Downloads a package; using bk or ftp; opens it in the with-external-packages-dir directory'''
    self.framework.log.write('Downloading '+self.name+'\n')
    import urllib

    packages  = self.framework.argDB['with-external-packages-dir']
    tarname   = self.name+'.tar'
    tarnamegz = tarname+'.gz'
    try:
      urllib.urlretrieve(self.download[0], os.path.join(packages,tarnamegz ))
    except Exception, e:
      raise RuntimeError('Error downloading '+self.name+': '+str(e))
    try:
      config.base.Configure.executeShellCommand('cd '+packages+'; gunzip '+tarnamegz, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error unzipping '+tarname+': '+str(e))
    try:
      config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf '+tarname, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error doing tar -xf '+tarname+': '+str(e))
    os.unlink(os.path.join(packages, tarname))
    self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+'\n')

  def getDir(self):
    '''Find the directory containing the package'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    Dir = None
    for dir in os.listdir(packages):
      if dir.startswith(self.name) and os.path.isdir(os.path.join(packages, dir)):
        Dir = dir
    if Dir is None:
      self.framework.log.write('Did not located already downloaded '+self.name+'\n')
      raise RuntimeError('Error locating '+self.name+' directory')
    return os.path.join(packages, Dir)

  def configure(self):
    '''Determines if the package should be configured for, then calls the configure'''
    if self.framework.argDB['download-'+self.package]:
      self.framework.argDB['with-'+self.package] = 1
    if 'with-'+self.package+'-dir' in self.framework.argDB or 'with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
      
    if self.framework.argDB['with-'+self.package]:
      if hasattr(self,'mpi') and self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.framework.argDB['with-64-bit-ints']:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')    
      if not self.clanguage.precision.lower() == 'double':
        raise RuntimeError('Cannot use '+self.name+' withOUT double precision numbers, it is not coded for this capability')    
      if not self.complex and self.clanguage.scalartype.lower() == 'complex':
        raise RuntimeError('Cannot use '+self.name+' with complex numbers it is not coded for this capability')    
      self.executeTest(self.configureLibrary)
    return

