# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.7.2.dev0'

from src_others.CTGAN.ctgan.demo import load_demo
from src_others.CTGAN.ctgan.synthesizers.ctgan import CTGAN
from src_others.CTGAN.ctgan.synthesizers.tvae import TVAE

__all__ = (
    'CTGAN',
    'TVAE',
    'load_demo'
)
