from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Fallback to hardcoded requirements
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "networkx>=2.6",
        "matplotlib>=3.5.0",
        "graphviz>=0.20.0",
        "fuzzywuzzy>=0.18.0",
        "python-levenshtein>=0.12.0",
    ]

setup(
    name="adel-lite",
    version="0.1.2",
    author="Parth Nuwal",
    author_email="parthnuwal7@gmail.com",
    description="Automated Data Elements Linking - Lite: Schema generation and profiling for Pandas DataFrames",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Parthnuwal7/adel-lite.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "adel-lite=adel_lite.cli:main",
        ],
    },
)
