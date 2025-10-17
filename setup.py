from setuptools import setup, find_packages

setup(
    name='vcat',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # Add dependencies here
    description='VLBI Comprehensive Analysis Toolkit',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/mpifr-vlbi/VCAT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
