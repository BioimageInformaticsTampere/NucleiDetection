from setuptools import find_packages, setup

setup(
    name="nucleidetection",
    packages=find_packages(),
    version="0.1.0",
    description='Supplementary material for article "Generalized fixation invariant nuclei detection through domain adaptation based deep learning" by Valkonen et al.',
    author="Bioimage Informatics Tampere",
    license="",
    install_requires=[
        "numpy>=1.16.0",
        "scipy==1.4.1",
        "scikit-image==0.17.2",
        "matplotlib==3.3.1",
        "tensorflow==2.3.0",
    ],
)
