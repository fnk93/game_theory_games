import os
import sys

texfile1 = 'template.tex'
texvile2 = 'template2.tex'
f = open(texfile1, 'r+')
data = f.read()
f.close()
with open(texvile2, 'w') as texfile:
    texfile.write(data)
    texfile.write('\\\\')
    texfile.close()
