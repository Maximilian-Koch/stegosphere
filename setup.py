from setuptools import setup, find_packages

setup(
    name="stegosphere",
    version="1.2.2",
    description="A flexible steganography and steganalysis library supporting various file types, including encryption and compression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Maximilian Koch",
    url="https://github.com/Maximilian-Koch/stegosphere",
    packages=find_packages(include=["stegosphere", "stegosphere.*"]),
    include_package_data=True,
    install_requires=[
        "numpy"
    ],
    license="GPL-3.0-or-later",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
