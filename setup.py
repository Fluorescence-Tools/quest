#!/usr/bin/python
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from distutils.core import setup
from Cython.Build import build_ext
import numpy


def make_extension(ext):
    """generate an Extension object from its dotted name
    """
    name = (ext[0])[2:-4]
    name = name.replace("/", ".")
    name = name.replace("\\", ".")
    sources = ext[0:]
    return Extension(
        name,
        sources=sources,
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=[],
        extra_link_args=[],
        libraries=[],
        library_dirs=["."],
        language="c++")


args = sys.argv[1:]
# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")


# and build up the set of Extension objects
extension_list = [
    ['./quest/lib/fps/fps.pyx', './quest/lib/fps/mt19937cok.cpp'],
    ['./quest/lib/math/functions/rdf.pyx'],
    ['./quest/lib/math/functions/small.pyx'],
    ['./quest/lib/math/linalg/vector.pyx'],
    ['./quest/lib/structure/cStructure.pyx'],
    ['./quest/lib/tools/dye_diffusion/photon.pyx', './quest/lib/tools/dye_diffusion/mt19937cok.cpp'],
]

extensions = [make_extension(extension) for extension in extension_list]

long_description = "QuEst is a Quenching Estimator"
setup(
    version="19.8.13",
    description="QuEst estimates the fluorescence quenching of dyes tethered to proteins",
    long_description=long_description,
    author="Thomas-Otavio Peulen",
    author_email='thomas.otavio.peulen@gmail.com',
    url='https://github.com/Fluorescence-Tools/quest',
    name="quest",
    classifiers=[
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
    keywords='fluorescence quenching',
    packages=find_packages(),
    package_data={
        '': ['*.json', '*.yaml', '*.ui', '*.png', '*.svg', '*.css', '*.so', '*.dll'],
    },
    install_requires=['numpy'],
    ext_modules=extensions,
    cmdclass={
        'build_ext': build_ext
    }
)

