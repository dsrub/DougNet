from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dougnet",
    version="0.1.5",
    author="Douglas Rubin",
    author_email="douglas.s.rubin@gmail.com",
    description="A lightweight, deep learning library written in pure Python for pedagogical purposes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsrub/DougNet",
    packages=find_packages(exclude=['examples', 'requirements']),
    install_requires=["numpy", "numba", "tqdm"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)