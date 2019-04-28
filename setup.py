import pathlib

import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="woodpecker-ml",
    version="0.1",
    description="A python library used for woodpecker structure interpretation.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/tlapusan/woodpecker",
    author="Tudor Lapusan",
    author_email="tudor.lapusan@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=setuptools.find_packages(),

)

if __name__ == "__main__":
    print("Asfda")
