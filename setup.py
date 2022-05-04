import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="subpixel",
    version="0.0.1",
    author="Rohit Rajesh, Adithya M",
    author_email="rajesh.rohit04@gmail.com, adithyamanjunatha01@gmail.com",
    description="Train Computer Vision models on your custom dataset with just a few lines of code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/04RR/Computer-Vision-Training-Library",
    project_urls={
        "Bug Tracker": "https://github.com/04RR/Computer-Vision-Training-Library/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
