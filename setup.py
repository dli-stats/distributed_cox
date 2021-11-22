"""Setup script."""

from setuptools import setup, find_packages

setup(
    name='distributed_cox',
    version='0.1dev',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'distributed_cmd=distributed_cox.distributed.cmd:main'
        ],
    },
    python_requires='>=3.6',
    install_requires=[
        'oryx==0.2.1',
        'jax==0.2.16',
        'jaxlib==0.1.68',
        'numpy',
        'pandas',
        'dataclasses_json',
        'tqdm',
        'sacred',
        'simpleeval',
        'frozendict',
    ],
    license='MIT license',
    long_description=open('README.md').read(),
)
