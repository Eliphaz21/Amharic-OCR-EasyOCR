from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="amharic-ocr-easyocr",
    version="1.0.0",
    author="Eliphaz21",
    author_email="eliphaz@users.noreply.github.com",
    description="End-to-end Amharic and Ge'ez Optical Character Recognition System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eliphaz21/Amharic-OCR-EasyOCR",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
