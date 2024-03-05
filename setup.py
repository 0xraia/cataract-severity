from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Cataract Severity Detection Interface'

# Setting up
setup(
    name="cataractseverity",
    version=VERSION,
    author="Mamang Joko",
    author_email="aimar.airdrop123@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['opencv-python', 'tensorflow', 'camconnect', 'numpy'],
    keywords=['python', 'interface'],
    license="MIT",
    url="https://github.com/0xraia/cataract-severity.git",
    project_urls={
        'Source': 'https://github.com/0xraia/cataract-severity.git',
    },
)
