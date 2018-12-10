import os
import re
import sys
import glob
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copymode
from setuptools import find_packages

setup(
    name='kgexpr',
    version='0.1.0',
    description='knowledge representation learning',
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url='http://github.com/fantasticfears/kgexpr',
    author='Erick Guan',
    author_email='fantasticfears@gmail.com',
    license='GPLv2',
    packages=find_packages(),
    install_requires=['kgekit', 'numpy>=1.10', 'pytorch>=0.4'],
    test_suite='tests',
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
)
