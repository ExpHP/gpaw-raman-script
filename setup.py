
import sys
from setuptools import setup

try:
    import gpaw
    assert gpaw.__version__ == '21.1.1b1'
except:
    print('Due to a reliance on a custom version of gpaw that must be manually built,', file=sys.stderr)
    print('this setup.py doesn\'t automatically handle dependencies.', file=sys.stderr)
    print(r'Please see:  https://gist.github.com/ExpHP/dce34c5008a0a1dffaf2bf2a1dfe7db9', file=sys.stderr)
    sys.exit(1)

setup(
    name='ep_script',
    version='0.0.1',
    description='GPAW Raman Script',
    author='Michael Lamparski',
    author_email='diagonaldevice@gmail.com',
    url='https://github.com/ExpHP/gpaw-raman-script',
    packages=['ep_script'],
)
