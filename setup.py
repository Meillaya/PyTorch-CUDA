import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_CXX_FLAGS=-fno-lto',
            '-DCMAKE_CUDA_FLAGS=-fno-lto',
            '-DPYBIND11_LTO_CXX_FLAGS=',
            '-DPYBIND11_LTO_LINKER_FLAGS=',
        ]

        build_args = []

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, 'parallel') and self.parallel:
                build_args += [f'-j{self.parallel}']

        # Check if we have ninja
        try:
            subprocess.run(['ninja', '--version'], check=True, stdout=subprocess.DEVNULL)
            cmake_args += ['-G', 'Ninja']
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Fall back to default generator

        # Find pybind11 cmake directory
        try:
            import pybind11
            pybind11_cmake_dir = pybind11.get_cmake_dir()
            cmake_args += [f'-DCMAKE_PREFIX_PATH={pybind11_cmake_dir}']
        except ImportError:
            pass

        # Ensure we have a build directory
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure
        subprocess.run(
            ['cmake', ext.sourcedir] + cmake_args, 
            cwd=build_temp, 
            check=True
        )

        # Build
        subprocess.run(
            ['cmake', '--build', '.'] + build_args, 
            cwd=build_temp, 
            check=True
        )


setup(
    name='pyminitorch',
    ext_modules=[CMakeExtension('pyminitorch')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
) 