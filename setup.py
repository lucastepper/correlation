#!/usr/bin/env python
from setuptools import setup


setup(
    name='correlation',
    version='0.1',
    description='Compute Corrs via FFT',
    author='Lucas Tepper',
    author_email='lucas.tepper.91@gmail.com',
          install_requires=[
          'mkl_fft',
      ],
    packages=[
        "correlation"
    ]
     )
