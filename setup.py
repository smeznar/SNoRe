import os
from setuptools import setup


with open("README.md", "r") as fl:
    long_description = fl.read()


def parse_requirements(file):
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


setup(name='snore-embedding',
      version='0.1',
      url='https://github.com/smeznar/SNoRe',
      author='Sebastian Mežnar and Blaž Škrlj',
      author_email='smeznar@gmail.com',
      license='GNU General Public License v3.0',
      keywords=['graph', 'representation learning', 'symbolic', 'snore', 'unsupervised learning'],
      description='SNoRe: Scalable Unsupervised Learning of Symbolic Node Representations',
      long_description=long_description,
      long_description_type='text/markdown',
      py_modules=['snore'],
      classifiers=['Intended Audience :: Information Technology',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      package_dir={'': 'src'},
      install_requires=parse_requirements('requirements.txt'))
