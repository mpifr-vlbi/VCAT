from setuptools import setup, find_packages

setup(
    name='vcat-vlbi',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # Add dependencies here
    description='VLBI Comprehensive Analysis Toolkit',
    author='Anne-Kathrin Baczko, Vieri Bartolini, Florian Eppel, Felix Pötzl, Luca Ricci, Jan Röder, Florian Rösch',
    author_email='anne-kathrin.baczko@chalmers.se, vbartolini@mpifr-bonn.mpg.de, florian@eppel.space, luca.ricci@uni-wuerzburg.de, jroeder@mpifr-bonn.mpg.de, florian.roesch@uni-wuerzburg.de',
    maintainer="Florian Eppel"
    mainter_email="florian@eppel.space"
    url='https://github.com/mpifr-vlbi/VCAT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    license='GPLv3',
    keywords='VLBI astronomy analysis radio-astronomy radio agn jets'
    project_urls={
        "Documentation": "https://github.com/mpifr-vlbi/VCAT/wiki",
        "Bug Tracker": "https://github.com/mpifr-vlbi/VCAT/issues",
    },
)
