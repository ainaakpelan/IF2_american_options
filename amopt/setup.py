from setuptools import setup, find_packages

setup(
    name="amopt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn"],
    author="Filip Tomaszewski",
    description="American Option Pricing Methods",
    python_requires=">=3.7",
)
