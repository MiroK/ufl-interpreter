#!/usr/bin/env python

from distutils.core import setup

setup(name = 'iufl',
      version = '0.1',
      description = 'Enable evaluation of UFL expressions',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/iufl.git',
      packages = ['iufl'],
      package_dir = {'iufl': 'iufl'}
)
