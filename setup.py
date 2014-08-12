from distutils.core import setup

setup(
    name='CrKr',
    version='0.1.0',
    author='Rui Silva',
    author_email='rui.teixeira.silva@ist.utl.pt',
    packages=['CrKr', 'CrKr.tests'],
    url='http://pypi.python.org/pypi/CrKr/',
    license='LICENSE.txt',
    description='Python implementation of the CrKr algorithm.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.6.2"
    ],
)
