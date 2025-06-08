from setuptools import setup, find_packages

setup(
    name="mvcp",
    version="0.1.0",
    description="Model for 5P Change Points - Temperature with energy consumption analysis tool",
    author="Rundong Liao",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "openpyxl",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
