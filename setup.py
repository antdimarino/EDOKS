import os
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

with open("requirements.txt") as f:
    required = f.read().splitlines()

class CMakeBuild(build_ext):
    def run(self):
        # Controlla che CMake sia disponibile
        if not self.cmake_available():
            raise RuntimeError("CMake non trovato. Installalo prima di procedere.")

        # Percorso alla directory del codice C
        cmake_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "edoks/emd_c"))
        build_temp = os.path.abspath(self.build_temp)
        build_lib = os.path.abspath(self.build_lib)

        os.makedirs(build_temp, exist_ok=True)

        # Configura e compila con CMake
        subprocess.check_call(
            [
                "cmake",
                cmake_source_dir,
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_lib}",
                "-DCMAKE_BUILD_TYPE=Release"
            ],
            cwd=build_temp
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)

        for filename in os.listdir(build_lib):
            if filename.endswith(".so") or filename.endswith(".dll"):
                shutil.copy(os.path.join(build_lib, filename), os.path.join("edoks", filename))

    def cmake_available(self):
        try:
            subprocess.check_output(["cmake", "--version"])
            return True
        except OSError:
            return False

setup(
    name="edoks",
    version="0.0.1",
    url="https://github.com/user/EDOK",
    author=['Bevilacqua Vincenzo', 'Di Marino Antonio'],
    author_email=['vincenzo.bevilacqua@icar.cnr.it', 'antonio.dimarino@icar.cnr.it'],
    description="A library for compute EDOKS metric.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["edoks", "edoks.emd_c"],
    data_files=[('edoks.emd_c', ['edoks/emd_c/libemd.so'])],
    include_package_data = True,
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
