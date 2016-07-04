#!/usr/bin/env python 

# import os
from setuptools import  find_packages

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

ext_modules = []
ext_modules += [
    Extension("pyrec.evaluate.cy_ranking_metric", [
              "pyrec/evaluate/cy_ranking_metric.pyx"]),
    Extension("pyrec.utils.data_utils.data_cython_helpers", [
          "pyrec/utils/data_utils/data_cython_helpers.pyx"]),
]

setup(
    name="pyrec",
    version="0.1",
    description="Pyrec - Python recommender engine for implicit feedback datasets",
    author="Suvash Sedhain",
    author_email="mesuvash@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_dir={'': '.'},
    ext_modules=cythonize(ext_modules),
)
