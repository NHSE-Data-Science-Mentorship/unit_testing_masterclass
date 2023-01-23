from setuptools import setup, find_packages


setup(
    name="learn_unit_testing",
    tests_require=["pytest"],
    version="0.1",
    packages=find_packages(include=["src/learn_unit_testing"]),
    package_dir={"": "src"},
)
