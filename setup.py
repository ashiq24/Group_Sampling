from setuptools import setup, find_packages


setup(
    name='gsampling',
    version='0.1.0',
    author='Md Ashiqur Rahman',
    author_email='rahman79@purdue.edu', 
    description='Group-equivariant sampling methods for PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ashiq24/Group_Sampling',
    packages=find_packages(include=['gsampling', 'gsampling.*']),
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.8',
    keywords=[
        'deep-learning',
        'pytorch',
        'group-equivariant',
        'sampling-methods'
    ],
)
