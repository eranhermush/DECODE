from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DECODE',
    version='0.1.0',
    author='Eran Hermush and Roded Sharan',
    description='Deconvolution of Bulk Gene Expression Data into Cell Fractions',
    url='https://github.com/eranhermush/DECODE',
    packages=find_packages('DECODE'),
    package_dir={'': 'DECODE'},
    install_requires=requirements
)