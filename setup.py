from setuptools import setup, find_packages

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="high-performance-python-lbm",
    version="0.1.0",
    author="jonas",
    author_email="jonas.burster@gmail.com",
    url="https://github.com/jonas27/high-performance-python-lbm",
    packages=find_packages(),
    python_requires=">=3.6",
    platforms=["Linux"],
    install_requires=REQUIREMENTS,
    include_package_data=True,
)