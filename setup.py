from setuptools import setup, find_packages

setup(
    name='xami_model',
    version='0.1',
    packages=find_packages(), 
    install_requires=[
        
    ],
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.png', '*.jpg'], 
    },
    description='XAMI: XMM-Newton optical Artefact Mapping for astronomical Instance segmentation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Elisabeta-Iulia Dima and ESA contributors',
    author_email='iuliaelisa15@yahoo.com',
    url='https://github.com/ESA-Datalabs/XAMI-model',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
