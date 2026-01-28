import os
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths

MODULE_NAME = "nv_accel"

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")


def pkg_config(package, flag):
    try:
        output = subprocess.check_output(["pkg-config", flag, package]).decode().strip()
        if not output:
            return []
        return output.split()
    except subprocess.CalledProcessError:
        return []


# Gather FFmpeg flags
ffmpeg_packages = ["libavcodec", "libavformat", "libavutil"]
include_dirs = []
library_dirs = []
libraries = ["nppicc", "nppig", "nppc"]  # NPP libraries

for pkg in ffmpeg_packages:
    cflags = pkg_config(pkg, "--cflags")
    libs = pkg_config(pkg, "--libs")

    for flag in cflags:
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])

    for flag in libs:
        if flag.startswith("-L"):
            library_dirs.append(flag[2:])
        elif flag.startswith("-l"):
            libraries.append(flag[2:])

# Remove duplicates
include_dirs = list(set(include_dirs))
library_dirs = list(set(library_dirs))
libraries = list(set(libraries))

rpath_dirs = []
for p in library_paths():
    if p not in rpath_dirs:
        rpath_dirs.append(p)

extra_link_args = []
for p in rpath_dirs:
    extra_link_args.append(f"-Wl,-rpath,{p}")

print(f"Include dirs: {include_dirs}")
print(f"Library dirs: {library_dirs}")
print(f"Libraries: {libraries}")

setup(
    name=MODULE_NAME,
    ext_modules=[
        CUDAExtension(
            MODULE_NAME,
            [
                "src/Decoder.cpp",
                "src/Encoder.cpp",
                "src/rgb_to_nv12.cu",
                "./bindings.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-O3"]},
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
