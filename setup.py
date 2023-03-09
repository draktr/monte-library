from setuptools import setup

setup(
    name="carlo",
    version="0.1.0",
    description="Carlo is a set of Monte Carlo methods in Python. The package is written to be flexible, clear to understand and encompass variety of Monte Carlo methods.",
    url="https://github.com/draktr/carlo",
    author="draktr",
    license="MIT License",
    packages=["carlo"],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
    ],
)
