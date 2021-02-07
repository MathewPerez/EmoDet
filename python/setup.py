from setuptools import find_packages, setup
setup(
    name='emodet',
    packages=find_packages(),
    version='0.1.0',
    description='API for BERT Model trained to detect emotions of text',
    author='Mathew Perez',
    license='Apache 2.0',
    install_requires=['numpy', 'torch', 'transformers'],
)