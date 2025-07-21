from setuptools import setup, find_packages

setup(
    name="stegosphere",
    version="1.3",
    description="A flexible steganography and steganalysis library supporting various file types",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Maximilian Koch",
    url="https://github.com/Maximilian-Koch/stegosphere",
    project_urls={'Documentation':'https://maximilian-koch.github.io/stegosphere/',
                    'Source':'https://github.com/Maximilian-Koch/stegosphere'},
    packages=find_packages(include=["stegosphere", "stegosphere.*"]),
    include_package_data=True,
    install_requires=[
    "numpy",
    "lazy-loader"
    ],
    extra_requires={'complete':[
                    "pillow",
                    "opencv-python",
                    "cryptography",
                    "soundfile",
                    "matplotlib",
                    "fonttools",
                    "scipy"
                ]},
    license="GPL-3.0-or-later",
    python_requires=">=3.6",
    classifiers = [
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Programming Language :: C",
    "Topic :: Communications",
    "Topic :: Communications :: File Sharing",
    "Topic :: Multimedia",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Security",
    "Topic :: Security :: Cryptography"],
)