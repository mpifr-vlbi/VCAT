from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='vcat-vlbi',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.1",
        "astropy>=6.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "scikit-image>=0.25.0",  
        "sympy>=1.12",
        "pexpect>=4.8",
        "astroquery>=0.4.6",
        "tqdm>=4.66",
        "colormaps>=0.1"
        ],
    description='VLBI Comprehensive Analysis Toolkit',
    author='Anne-Kathrin Baczko, Vieri Bartolini, Florian Eppel, Felix Pötzl, Luca Ricci, Jan Röder, Florian Rösch',
    author_email='anne-kathrin.baczko@chalmers.se, vbartolini@mpifr-bonn.mpg.de, florian@eppel.space, luca.ricci@uni-wuerzburg.de, jroeder@mpifr-bonn.mpg.de, florian.roesch@uni-wuerzburg.de',
    maintainer="Florian Eppel",
    maintainer_email="florian@eppel.space",
    url='https://github.com/mpifr-vlbi/VCAT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    license='GPLv3',
    keywords='VLBI astronomy analysis radio-astronomy radio agn jets',
    project_urls={
        "Documentation": "https://github.com/mpifr-vlbi/VCAT",
        "Bug Tracker": "https://github.com/mpifr-vlbi/VCAT",
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
