from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(name='fashionStyleClassification',
version='1.0.0',
description='woman fashionstyleclassification network',
url='https://github.com/pajamacoders/fashionStyleClassification/tree/dev_packaging',
packages=find_packages(),
package_data={'fashionStyleClassification':['config/*/*.yaml']},
include_package_data=True,
author='pajamacoders',
author_email='codespectator@gmail.com',
license='pajamacoders',
zip_safe=False,
python_requires='>=3.5',
install_requires=requirements)