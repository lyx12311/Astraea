from setuptools import setup

with open("Astraea/version.py", "r") as f:
    exec(f.read())

setup(name='Astraea',
      version=__version__,
      description='Predict rotation period of a star from various features',
      url='https://github.com/lyx12311/Astraea',
      author='Yuxi(Lucy) Lu',
      author_email='lucylulu12311@gmail.com',
      license=' ',
      packages=['Astraea'],
      install_requires=['numpy', 'pandas', 'astropy', 'sklearn','matplotlib'],
      zip_safe=False,
      include_package_data=True
      )
