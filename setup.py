import os
import platform
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as error:
            raise RuntimeError(
                "CMake must be installed to build flash-radiomics"
            ) from error

        self.cuda_available = self._resolve_cuda_build()
        status = "enabled" if self.cuda_available else "disabled; CPU remains enabled"
        print(f"CUDA backend: {status}")
        for extension in self.extensions:
            self.build_extension(extension)

    @staticmethod
    def _nvcc_available():
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
        if result.returncode != 0:
            return False
        if "release" in result.stdout.lower():
            version = result.stdout.lower().split("release", 1)[1].split(",", 1)[0]
            print(f"Detected CUDA:{version}")
        return True

    def _resolve_cuda_build(self):
        """Resolve FLASH_RADIOMICS_ENABLE_CUDA as auto, on, or off."""
        requested = os.environ.get("FLASH_RADIOMICS_ENABLE_CUDA", "auto").lower()
        aliases = {
            "1": "on",
            "true": "on",
            "yes": "on",
            "0": "off",
            "false": "off",
            "no": "off",
        }
        requested = aliases.get(requested, requested)
        if requested not in {"auto", "on", "off"}:
            raise RuntimeError(
                "FLASH_RADIOMICS_ENABLE_CUDA must be auto, on, or off"
            )
        if requested == "off":
            return False
        available = self._nvcc_available()
        if requested == "on" and not available:
            raise RuntimeError(
                "CUDA was required but nvcc is unavailable. Install the CUDA "
                "toolkit or set FLASH_RADIOMICS_ENABLE_CUDA=off."
            )
        return available

    def build_extension(self, extension):
        extension_directory = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(extension.name))
        )
        build_directory = Path(self.build_temp)
        build_directory.mkdir(parents=True, exist_ok=True)
        native_cpu = os.environ.get("FLASH_RADIOMICS_NATIVE_CPU", "").lower()
        native_cpu_enabled = native_cpu in {"1", "on", "true", "yes"}

        cmake_arguments = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_directory}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DFLASH_RADIOMICS_NATIVE_CPU={'ON' if native_cpu_enabled else 'OFF'}",
            f"-DENABLE_CUDA={'ON' if self.cuda_available else 'OFF'}",
        ]
        build_arguments = ["--config", "Release"]

        cpu_mode = "native" if native_cpu_enabled else "portable"
        print(f"CPU build mode: {cpu_mode}")
        if platform.system() == "Darwin":
            python_architecture = platform.machine().lower()
            if python_architecture in {"arm64", "x86_64"}:
                cmake_arguments.append(
                    f"-DCMAKE_OSX_ARCHITECTURES={python_architecture}"
                )
            deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
            cmake_arguments.append(
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"
            )
        if platform.system() != "Windows":
            build_arguments += ["--", "-j4"]

        environment = os.environ.copy()
        environment["CXXFLAGS"] = (
            f"{environment.get('CXXFLAGS', '')} "
            f'-DVERSION_INFO="{self.distribution.get_version()}"'
        )
        print("Configuring native extension")
        subprocess.check_call(
            ["cmake", extension.sourcedir] + cmake_arguments,
            cwd=self.build_temp,
            env=environment,
        )
        print("Building native extension")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_arguments,
            cwd=self.build_temp,
        )


readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)


setup(
    name="flash-radiomics",
    version="1.0.0",
    author="Shanli Ding",
    author_email="sding2@mdanderson.org",
    description="Performance-optimized radiomics feature extraction with CPU and GPU acceleration",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["flash_radiomics", "flash_radiomics.*"]),
    include_package_data=False,
    package_data={
        "flash_radiomics": [
            "schemas/*.py",
            "schemas/*.yaml",
        ],
    },
    ext_modules=[
        CMakeExtension("flash_radiomics._native", sourcedir=str(Path(__file__).parent))
    ],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[],
    python_requires=">=3.10,<3.15",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: C",
    ],
    zip_safe=False,
)
