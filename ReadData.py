# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:29:38 2016

@author: kevin

TODO: load spectrum data from .*argus files 
"""
#folder in PC
filespath = 'D:\Code\Matlab\SpectrumLearning\Data\OriginData\1710-1960\20151216\02\201512151900.argus'
try:
  files=open(filespath,'rb')
  content=files.read()
finally:
  if files:
    files.close()
