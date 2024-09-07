from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dougnet",
    version="0.1.0",
    author="Douglas Rubin",
    author_email="douglas.s.rubin@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/dsrub/dougnet",
    packages=find_packages(exclude=['examples', 'requirements']),
    install_requires=["numpy==1.24.4", "numba==0.58.1", "tqdm==4.66.5"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)