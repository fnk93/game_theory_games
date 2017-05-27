import numpy as np
from copy import deepcopy
from GameTheory_Game.Solving_Methods import *
from django.http import HttpResponse
from django.template import Context
from django.template.loader import get_template
from subprocess import Popen, PIPE
import tempfile
import os

set = {'csrfmiddlewaretoken': ['TP6khmBFjWmF47MxMYC0jl1B5wGZxK5uqXyLv8P6T4kl93Fhg7pEX7Nt3yS2t2Dx'], 'mat1': ['[[ -2. -5. 0.]\r\n [ -1. 10. 10.]\r\n [ 7. -6. 7.]]'], 'mat2': ['[[ 2. 5. 0.]\r\n [ 1. -10. -10.]\r\n [ -7. 6. -7.]]'], 'mod': ['0'], 'ba1': ['1'], 'ba2': ['0'], 'pdf': ['PDF-Datei herunterladen']}
print(np.array(set['mat1'][0]))
print(set['mod'][0])
x = np.array(set['mat1'][0])
x.flatten()
y = list(set['mat1'])
print(y)
print(np.asarray(y))

test = "".join(y)
test = test.replace('.', '')
test = test.replace('[', '')
test = test.split(' ')
new_test = []
for str in test:
    new_test.append(str.replace("\r\n", ''))
print(new_test)
print(test)
print('!!!!!')
new_arr = []
temp = []
firstDigit = False
endFound= False
for i in range(len(new_test)):
    print(new_test[i])
    if "]" in new_test[i]:
        new_test[i] = new_test[i].replace("]", "")
        endFound = True
    if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
        firstDigit = True
        temp.append(int(new_test[i]))
    if (new_test[i] == "[" and temp and firstDigit) or endFound:
        new_arr.append(deepcopy(temp))
        print(temp)
        print('added')
        firstDigit = False
        endFound = False
        temp.clear()

print('bereinigtes Array')
print(new_arr)
print(np.asarray(new_arr).shape[0])
z = list(set['mat2'])
print(set['mat2'])
test2 = "".join(z)
test2 = test2.replace('.', '')
test2 = test2.replace('[', '')
test2 = test2.split(' ')
new_test2 = []
for str2 in test2:
    new_test2.append(str2.replace("\r\n", ''))

firstDigit = False
endFound = False
new_arr2 = []
temp2 = []
print(new_test2)
for j in range(len(new_test2)):
    print(new_test2[j])
    if "]" in new_test2[j]:
        endFound = True
        new_test2[j] = new_test2[j].replace("]", "")
    if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
        firstDigit = True
        temp2.append(int(new_test2[j]))
    if (new_test2[j] == "[" and temp and firstDigit) or endFound:
        new_arr2.append(deepcopy(temp2))
        print(temp2)
        temp2.clear()
        print('added')
        firstDigit = False
        endFound = False
print(new_arr2)

new_arr = np.asarray(new_arr)
new_arr2 = np.asarray(new_arr2)
print(new_arr)
print(new_arr2)
print('shape')
print(new_arr2.shape[1])
print(get_calculations_latex(new_arr, new_arr2, True, int(set['ba1'][0]), int(set['ba2'][0]), 1)[0])
print(get_calculations_latex(new_arr, new_arr2, True, int(set['ba1'][0]), int(set['ba2'][0]), 1)[1])
print()
print(get_calculations_latex(new_arr, new_arr2, True, int(set['ba1'][0]), int(set['ba2'][0]), 1)[2])

mfb = r'abc\abc'
print(mfb)


# texfile1 = 'template.tex'
# texvile2 = 'template2.tex'
# f = open(texfile1, 'r+')
# data = f.read()
# f.close()
# with open(texvile2, 'w') as texfile:
#     texfile.write(data)
#     texfile.write('\\\\')
#     texfile.close()
# context = Context(get_calculations_latex(new_arr, new_arr2, True, int(set['ba1'][0]), int(set['ba2'][0]), 1)[2])
# template = get_template('template3.tex')
# rendered_tpl = template.render(context).encode('utf-8')
# with tempfile.TemporaryDirectory() as tempdir:
#     for i in range(2):
#         process = Popen(
#             ['pdflatex', '-output-directory', tempdir],
#             stdin = PIPE,
#             stdout = PIPE
#         )
#         process.communicate(rendered_tpl)
#     with open(os.path.join(tempdir, 'texput.pdf'), 'rb') as f:
#         pdf = f.read()
