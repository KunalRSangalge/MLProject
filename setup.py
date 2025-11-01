from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(path:str)->List[str]:
    '''
    this function returns list of requirements
    '''
    requirements=[]
    with open(path) as obj:
        requirements = obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='MLProject',
    version='0.0.1',
    author='kunal',
    author_email='kunal.sangalge@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)