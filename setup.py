from setuptools import setup

with open("README.md", "r") as f:
    desc = f.read()

setup(
    name="monte-library",
    version="0.2.0",
    description="monte-library is a set of Monte Carlo methods in Python. The package is written to be flexible, clear to understand and encompass variety of Monte Carlo methods.",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/draktr/monte-library",
    author="draktr",
    license="MIT License",
    packages=["monte"],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
    ],
    keywords="montecarlo monte carlo, optimization, integration, sampling, mcmc, hmc, simulation, modelling",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    project_urls={
        "Documentation": "https://monte-library.readthedocs.io/en/latest/",
        "Issues": "https://github.com/draktr/monte-library/issues",
    },
)
