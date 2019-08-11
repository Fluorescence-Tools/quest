## It seems that the building process has problems with anaconda-accelerate
## Thus I only use the normal anaconda distribution
import os
import sys
import fnmatch
from distutils.extension import Extension
import shutil
import subprocess
import site
from Cython.Distutils import build_ext
from cx_Freeze import setup, Executable

import numpy
import zmq
import zmq.libzmq


libIncludeDir = "."
python_dir = 'C:\\Anaconda'
package_dir = site.getsitepackages()[1]
target_dir = 'E:\\141216\\build'
setup_dir = 'E:\\141216\\setup'

# Clear the target dir
shutil.rmtree(target_dir, ignore_errors=True)

# Process the includes, excludes and packages first
include_files = []
includes = ['numpy', 'matplotlib', 'matplotlib.backends.backend_qt4agg', 'sip', 'PyQt4.QtGui',
            'zmq', 'zmq.utils.garbage', 'zmq.backend.cython']
excludes = ['_gtkagg', '_tkagg', 'bsddb', 'curses', 'email',
            'tcl', 'Tkconstants', 'Tkinter', 'boto']
packages = ['scipy', 'scipy.special', 'scipy.optimize', 'scipy.stats',
            'OpenGL', 'sympy', 'emcee', 'mdtraj', 'IPython', 'pygments', 'zmq',
            'numpy', 'numpy.linalg']
path = []
dll_excludes = ['libgdk-win32-2.0-0.dll', 'libgobject-2.0-0.dll', 'tcl84.dll',
                'tk84.dll']


def makeExtension(ext):
    """generate an Extension object from its dotted name
    """
    name = (ext[0])[2:-4]
    name = name.replace("/", ".")
    name = name.replace("\\", ".")
    sources = ext[0:]
    return Extension(
        name,
        sources=sources,
        include_dirs=[libIncludeDir, numpy.get_include(), "."],  # adding the '.' to include_dirs is CRUCIAL!!
        # extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
        libraries=[],
        library_dirs=[libIncludeDir, "."],
        language="c++")


# and build up the set of Extension objects
eList = [['./mfm/fluorescence/tcspc/_tcspc.pyx'],
         ['./mfm/fluorescence/fcs/_fcs.pyx'],
         ['./mfm/math/reaction/_reaction.pyx'],
         ['./mfm/fluorescence/fps/_fps.pyx', './mfm/math/rand/mt19937cok.cpp'],
         ['./mfm/io/_tttrlib.pyx'],
         ['./mfm/math/linalg/_vector.pyx'],
         ['./mfm/math/functions/_special.pyx', './mfm/math/rand/mt19937cok.cpp'],
         ['./mfm/math/rand/_rand.pyx', './mfm/math/rand/mt19937cok.cpp'],
         ['./mfm/structure/cStructure.pyx'],
         ['./mfm/structure/potential/cPotentials.pyx'],
         ['./mfm/fluorescence/simulation/_photon.pyx', './mfm/math/rand/mt19937cok.cpp']
]

extensions = [makeExtension(extension) for extension in eList]
for e in extensions:
    e.pyrex_directives = {"boundscheck": False,
                          "wraparound": False,
                          "cdivision": True,
                          "profile": False
    }

include_files = []
for root, dirnames, filenames in os.walk('mfm'):
    for filename in fnmatch.filter(filenames, '*.ui') + fnmatch.filter(filenames, '*.dll'):
        include_files.append(os.path.join(root, filename))

for root, dirnames, filenames in os.walk('settings'):
    for filename in fnmatch.filter(filenames, '*.json'):
        include_files.append(os.path.join(root, filename))


# The setup for cx_Freeze is different from py2exe. Here I am going to
# use the Python class Executable from cx_Freeze
base = None
if sys.platform == "win32":
    base = "Win32GUI"

GUI2Exe_Target = Executable(
    # what to build
    script="mfm_gui.py",
    initScript=None,
    base=base,
    targetDir=target_dir,
    targetName="ChiSurf.exe",
    copyDependentFiles=True,
    icon='.\\mfm\\ui\\icons\\kitesurf.ico'
)

import glob2, scipy, guidata, guiqwt, sympy, emcee, matplotlib, IPython

explore_dirs = [os.path.dirname(numpy.__file__), os.path.dirname(scipy.__file__),
                os.path.dirname(guidata.__file__), os.path.dirname(guiqwt.__file__),
                os.path.dirname(sympy.__file__), os.path.dirname(emcee.__file__),
                os.path.dirname(matplotlib.__file__),
                os.path.dirname(IPython.__file__),
                os.path.dirname(zmq.__file__)
]

files = []
for d in explore_dirs:
    files.extend(glob2.glob(os.path.join(d, '*', '*.pyd')))

# Now we have a list of .pyd files; iterate to build a list of tuples into
# include files containing the source path and the basename
print("Including scipy and numpy files")
for f in files:
    fn = f.split(package_dir, 1)[1].replace('\\', '.').split('.pyd', 1)[0]
    includes.append(fn[1:])

# That's serious now: we have all (or almost all) the options cx_Freeze
# supports. I put them all even if some of them are usually defaulted
# and not used. Some of them I didn't even know about.
# http://www.riverbankcomputing.com/pipermail/pyqt/2013-June/032883.html

setup(
    version="0.0.2",
    description="Fluorescence-Fitting",
    author="Thomas Peulen",
    name="ChiSurf",
    ext_modules=extensions,
    cmdclass={
        'build_ext': build_ext
    },
    options={
        "build_exe": {"include_files": include_files,
                      "includes": includes,
                      "excludes": excludes,
                      "packages": packages,
                      "path": path,
                      'build_exe': target_dir,
                      'copy_dependent_files': True,
                      'optimize': 1,  # use '1' if docstrings are stripped (too much optimization) it won't work,
                      'compressed': True
        }
    },
    executables=[GUI2Exe_Target]
)

# This is a place where any post-compile code may go.
# You can add as much code as you want, which can be used, for example,
# to clean up your folders or to do some particular post-compilation
# actions.
if sys.platform.startswith('win'):
    shutil.copytree(package_dir + '\\guidata\\', target_dir + '\\guidata')
    shutil.copytree(package_dir + '\\zmq\\', target_dir + '\\zmq')
    shutil.copytree(package_dir + '\\pymol\\', target_dir + '\\pymol')
    shutil.copytree(package_dir + '\\pymol2\\', target_dir + '\\pymol2')
    shutil.copytree(package_dir + '\\guiqwt\\', target_dir + '\\guiqwt')
    shutil.copytree('.\\mfm\\ui\\fortune\\', target_dir + '\\mfm\\ui\\fortune\\')

    # Compress the files if UPX is found on the system path
    subprocess.call(['upx', target_dir+'\\*.*'])

    #Make an installer using InnoSetup
    subprocess.call(['C:\Program Files (x86)\Inno Setup 5\Compil32.exe', '/cc', 'setup.iss'])
    #Rename the installer to include the ChiSurf version
    old_name = os.path.join(setup_dir, 'chiSurf_setup.exe')
    import mfm
    new_name = os.path.join(setup_dir, 'chiSurf_setup-' + mfm.__version__ + '.exe')
    if os.path.exists(new_name):
        os.remove(new_name)
    os.rename(old_name, new_name)

# Clear the target dir
shutil.rmtree(target_dir, ignore_errors=True)
