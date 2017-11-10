# setup.py
from setuptools import setup, find_packages

setup(name='kerascNN',
  version='0.1',
  packages=find_packages(),
  description=' run MMNIST keras on gcloud ml-engine',
  author='Alan AN',
  author_email='alanan46@gmail.com',
  license='MIT',
  install_requires=[
      'keras',
      'scikit-learn>=0.18',
      'numpy',
      'h5py'
  ],
  zip_safe=False)
