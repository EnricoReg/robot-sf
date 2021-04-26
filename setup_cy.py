# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:26:16 2021

@author: Matteo Caruso
"""

from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
import os

EXCLUDE_FILES = []
#EXCLUDE_FILES = []

def get_ext_paths(root_dir, exclude_files):
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue
            if filename=='__init__.py':
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)

    return paths

setup(name='robot_sf',
      version='1.0.0',
      description='This package allows implementing a "gym-style" environment for the mobile robot navigating the crowd',
      url='https://github.com/matteocaruso1993/robot_env',
      author='Matteo Caruso and Enrico Regolin',
      author_email='matteo.caruso@phd.units.it',
      license="GPLv3",
      #packages=['robot_env'],
      packages=find_packages(),
      package_data = {'robot_sf': ['utils/maps/*.json','utils/config/map_config.toml']},
      install_requires=['numpy','Pillow','matplotlib','pysocialforce','python-math','jsons','toml','natsort'],
      zip_safe=False,
      ext_modules = cythonize(get_ext_paths('robot_sf', EXCLUDE_FILES), compiler_directives={'language_level':3}),
      include_package_data=True,
      python_requires='>=3.6'
      )
