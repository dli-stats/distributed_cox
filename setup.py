"""Setup script."""

from distutils.core import setup

setup(
    name='distributed_cox',
    version='0.1dev',
    packages=[
        'distributed_cox',
    ],
    entry_points={
        'console_scripts': ['distributed_cmd=distributed_cox.distributed.cmd:main'],
    },
    license='MIT license',
    long_description=open('README.md').read(),
)
