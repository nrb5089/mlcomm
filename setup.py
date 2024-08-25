from setuptools import setup, find_packages

setup(name='mlcomm',
      version='0.3',
      description='Package of ML algorithms for communication',
      url='https://github.gatech.edu/mb364/mlcomm',
      author='Nathan Blinn, Matthieu Bloch, Jana Boerger',
      author_email='nblinn6@gatech.edu',
      license='TBD',
      packages= find_packages(),
          install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        ],
      python_requires='>=3.10',
      zip_safe=False)
