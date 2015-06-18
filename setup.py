"""
A setuptools based setup module
"""

from setuptools import setup, find_packages

setup(
    name="anvil",
    version="0.1.0",
    description="A python project for sensor data analysis",
    author="Saeed Abdullah",
    author_email="me@saeedabdullah.com",

    # list of packages
    packages=['anvil'],

    install_requires=['pandas'],

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3 :: Only"
    ]
)
