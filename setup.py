from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="pylbm",
    version="0.1.1",
    author="jonas",
    author_email="jonas.burster@gmail.com",
    url="https://github.com/jonas27/pylbm",
    packages=find_packages(),
    python_requires=">=3.8",
    platforms=["Linux"],
    install_requires=REQUIREMENTS,
    include_package_data=True,
)
