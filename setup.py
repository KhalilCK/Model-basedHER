import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


extras = {
    'test': [
        'filelock',
        'pytest',
        'pytest-forked',
        'matplotlib',
        'pandas'
    ],
    'mpi': [
        'mpi4py'
    ]
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras['all'] = all_deps

setup(name='mher',
      packages=[package for package in find_packages()
                if package.startswith('mher')],
      install_requires=[
          'gym>=0.15.4, <0.16.0',
          'scipy',
          'tqdm',
          'joblib',
          'cloudpickle',
          'click',
      ],
      extras_require=extras,
      description='Model-based Hindsight Experience Replay.',
      version='0.1')


# ensure there is some tensorflow build with version above 1.4
import pkg_resources
tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version above 1.4'
from distutils.version import LooseVersion
assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version)) >= LooseVersion('1.4.0')
