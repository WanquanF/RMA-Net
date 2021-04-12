import os
import sys

os.chdir('./model/exfunc/expackages/point_masker')
os.system('python setup.py install')
os.chdir('../point_render')
os.system('python setup.py install')
