from setuptools import setup

setup(name='pyIPCA',
      version='1.0',
      description='python package for Incremental Prinicpal Component Analysis',
      author='Kevin Hughes',
      author_email='kevinhughes27@gmail.com',
      url='https://github.com/pickle27/pyIPCA',
      packages=['pyIPCA'],
      license='BSD License',
      install_requires=[
          'scikit-learn',
      ],
      platforms='Any',
      )