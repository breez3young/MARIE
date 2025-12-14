"""SMAX environment wrapper for MARIE"""
import os
# Force JAX to use CPU before importing SMAX
os.environ['JAX_PLATFORMS'] = 'cpu'

from env.smax.SMAX import SMAX

__all__ = ['SMAX']
