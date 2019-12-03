"""Setup script."""

from distutils.core import setup

setup(
    name='varderiv',
    version='0.1dev',
    packages=[
        'varderiv',
    ],
    entry_points={
        'console_scripts': ['distributed_cmd=varderiv.distributed.cmd:main'],
    },
    license='MIT license',
    long_description=open('README.md').read(),
)
