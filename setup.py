from setuptools import setup

setup(
    name='xmm_om_model',
    version='0.1.0',
    py_modules=['YoloSamPipeline'],
    install_requires=[
        
    ],  
    entry_points='''
        [console_scripts]
        example=example:example
    ''',
)