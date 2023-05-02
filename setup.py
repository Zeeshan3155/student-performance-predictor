from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path)->List[str]:
    hypen_e_dot = "-e ."
    with open(file_path) as fp:
        requirements = fp.readlines()
        requirements = [i.replace("\n","") for i in requirements]

    if hypen_e_dot in requirements:
        requirements.remove(hypen_e_dot)

    return requirements

setup(
    name="student_performance_predictor",
    version="0.0.1",
    author="Zeeshan Ahmed",
    author_email="zeeshanahmed4200@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements("requirements.txt")
    )