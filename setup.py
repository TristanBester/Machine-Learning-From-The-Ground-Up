from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
      name='mlgroundup',
      version='0.0',
      description='models from ground up',
      author='Tristan Bester',
      author_email='N/A',
      packages=['mlgroundup'],
      long_description=long_description,
      license="Not yet specified.",
      url='https://github.com/TristanBester/Machine-Learning-From-The-Ground-Up'
      )