import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(name='eao',
      version='0.2',
      description='Evolutionary Optimization',
      long_description=long_description,
      long_description_content_type='text/markdown',
      keywords=['evolution', 'evolutionary', 'algorithm', 'optimization', 'genetic'],
      author=u'Sascha M\u00fccke',
      author_email='sascha.muecke@tu-dortmund.de',
      url='https://github.com/smuecke/eao',
      license='GNU GPLv3',
      packages=setuptools.find_packages(),
      python_requires='>=3.5',
      install_requires=['numpy', 'tqdm'])
