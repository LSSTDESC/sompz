#!/usr/bin/env python
from __future__ import print_function
import os, re, sys

try:
    from setuptools import setup, find_packages
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    print('Unable to import setuptools.  Using distutils instead.')
    from distutils.core import setup
    # cf. http://stackoverflow.com/questions/37350816/whats-distutils-equivalent-of-setuptools-find-packages-python
    from distutils.util import convert_path
    def find_packages(base_path='.'):
        base_path = convert_path(base_path)
        found = []
        for root, dirs, files in os.walk(base_path, followlinks=True):
            dirs[:] = [d for d in dirs if d[0] != '.' and d not in ('ez_setup', '__pycache__')]
            relpath = os.path.relpath(root, base_path)
            parent = relpath.replace(os.sep, '.').lstrip('.')
            if relpath != '.' and parent not in found:
                # foo.bar package but no foo package, skip
                continue
            for dir in dirs:
                if os.path.isfile(os.path.join(root, dir, '__init__.py')):
                    package = '.'.join((parent, dir)) if parent else dir
                    found.append(package)
        return found
    import distutils
    print("Using distutils version",distutils.__version__)

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

packages = find_packages()
print('packages = ',packages)

dependencies = ['matplotlib', 'pandas', 'numpy', 'numba', 'h5py']

with open('README.md') as file:
    long_description = file.read()

# Read in the sompz version from sompz/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('sompz','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    sompz_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('sompz version is %s'%(sompz_version))

dist = setup(name="sompz",
      version=sompz_version,
      author="Chris Davis",
      author_email="chris.pa.davis@gmail.com",
      description="Self Organizing Maps for Photometric Redshifts",
      long_description=long_description,
      license = "BSD License",
      url="https://github.com/des-science/sompz",
      install_requires=dependencies,
      packages=packages,
      )
