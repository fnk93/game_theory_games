from gaming.Game import Game
from gaming.Solving_Methods import *
# from copy import deepcopy
from django.http import HttpResponse
from django.shortcuts import render
from gaming.postDataProcessing import reconstruct_matrix
from gaming.ZerosumGame import ZerosumGame


def game(request):
    if request.GET.get('generate'):
        new_game = ZerosumGame(maximum_int=10, minimum_int=-10, lin=np.random.randint(2, 5),
                               col=np.random.randint(2, 5))
        output2 = int(request.GET.get('spielart'))
        if int(request.GET.get('spielart')) == 0 or (2 <= int(request.GET.get('spielart')) <= 5):
            mode = 0
        else:
            mode = 1
        if int(request.GET.get('spielart')) == 2:
            new_game.matrix = np.asarray([[6,5,6,5],
                                          [1,4,2,-1],
                                          [8,5,7,5],
                                          [0,2,6,2]])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 3:
            new_game.matrix = np.asarray([[4,1,2],
                                          [1,5,0],
                                          [4,3,3],])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 4:
            new_game.matrix = np.asarray([[3,0,2],
                                          [-4,-1,-3],
                                          [2,-2,-1],])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 5:
            new_game.matrix = np.asarray([[4,5,2],
                                          [6,3,2],])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 6:
            new_game.matrix = np.asarray([[1,7,0,3],
                                          [0,0,3,5],
                                          [1,2,4,1],
                                          [6,0,2,0]])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 7:
            new_game.matrix = np.asarray([[8,6],
                                          [4,7],])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 8:
            new_game.matrix = np.asarray([[2,5],
                                          [4,3],
                                          [3,6],
                                          [5,4],
                                          [4,4]])
            new_game.matrix2 = new_game.matrix * -1
        elif int(request.GET.get('spielart')) == 9:
            new_game.matrix = np.asarray([[3,0,2],
                                          [4,5,1],
                                          [2,2,-1],])
            new_game.matrix2 = new_game.matrix * -1
        matrix1 = new_game.matrix
        matrix2 = new_game.matrix2
        fixed_strat_1 = np.random.randint(0, matrix1.shape[0])
        fixed_strat_2 = np.random.randint(0, matrix1.shape[1])
        aufgaben_pure = ['Maximin-Strategie(n) beider Spieler',
                         'Indeterminiertheitsintervall',
                         'Untere Spielwerte der Spieler',
                         'Garantiepunkt des Spiels',
                         'Bayes-Strategie von Spieler 1 gegenüber Strategie ' + str(fixed_strat_2 + 1) +
                         ' von Spieler 2',
                         'Bayes-Strategie von Spieler 2 gegenüber Strategie ' + str(fixed_strat_1 + 1) +
                         ' von Spieler 1',
                         'Gleichgewichtspunkt und dessen Auszahlungen']
        aufgaben_mixed = ['Gleichgewichtspunkt des Spiels',
                          'Strategienkombination im Gleichgewicht',
                          'Auszahlung für die Spieler']
        response = render(request, 'gaming/game.html', {'output': matrix1,
                                                         'output2': matrix2,
                                                         'game': new_game,
                                                         'modus': mode,
                                                         'work': aufgaben_pure,
                                                         'work_mixed': aufgaben_mixed,
                                                         'bay1': fixed_strat_2,
                                                         'bay2': fixed_strat_1,
                                                        'name': 'Spiel'})
    return response


def solution(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]

    data2 = request.POST['mat2']  # Matrix Spieler 2
    reconst2 = reconstruct_matrix(data2)
    data2 = reconst2[0]
    data3 = request.POST['mod']  #
    data3 = int(data3)
    data4 = request.POST['ba1']  #
    data4 = int(data4)
    data5 = request.POST['ba2']  #
    data5 = int(data5)
    data6 = str(request.POST)
    data7 = ''
    new_game2 = Game
    matrix_1 = np.asarray(reconst1[1])
    matrix_2 = np.asarray(reconst2[1])
    solution = ''
    solution = get_calculations_latex(matrix_1, matrix_2, zerosum=True, bay1=data4, bay2=data5, mode=data3)
    context = solution[2]
    context2 = solution[3]
    template = 'template.tex'
    response2 = render(request, template, context)

    response = render(request, 'gaming/solution.html', {'output': matrix_1,
                                                     'output2': matrix_2,
                                                     'modus': data3,
                                                     'bay1': data4,
                                                     'bay2': data5,
                                                     'sol': solution[0],
                                                     'dat': request.POST,
                                                        'name': 'Lösung',
                                                        'dict': context2})
    return response


def pdf_game(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]
    data2 = request.POST['mat2']  # Matrix Spieler 2
    reconst2 = reconstruct_matrix(data2)
    data2 = reconst2[0]
    data3 = request.POST['mod']  #
    data3 = int(data3)
    data4 = request.POST['ba1']  #
    data4 = int(data4)
    data5 = request.POST['ba2']  #
    data5 = int(data5)
    data6 = str(request.POST)
    data7 = ''
    new_game2 = Game
    matrix_1 = np.asarray(reconst1[1])
    matrix_2 = np.asarray(reconst2[1])
    solution = ''
    solution = get_calculations_latex(matrix_1, matrix_2, zerosum=True, bay1=data4, bay2=data5, mode=data3)
    context = solution[3]
    template = 'template.tex'
    response2 = render(request, template, context)
    #game_matrix1 = request.POST['beb']
    gaming = request.POST['game']

    if request.POST.get('pdf2'):
        aufgaben_pure = ['Maximin-Strategie(n) beider Spieler',
                         'Indeterminiertheitsintervall',
                         'Untere Spielwerte der Spieler',
                         'Garantiepunkt des Spiels',
                         'Bayes-Strategie von Spieler 1 gegenüber Strategie ' + str(
                             data4 + 1) + ' von Spieler 2',
                         'Bayes-Strategie von Spieler 2 gegenüber Strategie ' + str(
                             data5 + 1) + ' von Spieler 1',
                         'Gleichgewichtspunkt und dessen Auszahlungen']
        aufgaben_mixed = ['Gleichgewichtspunkt des Spiels',
                          'Strategienkombination im Gleichgewicht',
                          'Auszahlung für die Spieler']
        response = render(request, 'gaming/index.html', {'output': matrix_1,
                                                         'output2': matrix_2,
                                                         'modus': data3,
                                                         'bay1': data4,
                                                         'bay2': data5,
                                                         'work': aufgaben_pure,
                                                         'work_mixed': aufgaben_mixed,
                                                         'game': gaming,})
    return response


def pdf_solution(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]
    data2 = request.POST['mat2']  # Matrix Spieler 2
    reconst2 = reconstruct_matrix(data2)
    data2 = reconst2[0]
    data3 = request.POST['mod']  #
    data3 = int(data3)
    data4 = request.POST['ba1']  #
    data4 = int(data4)
    data5 = request.POST['ba2']  #
    data5 = int(data5)
    data6 = str(request.POST)
    data7 = ''
    new_game2 = Game
    matrix_1 = np.asarray(reconst1[1])
    matrix_2 = np.asarray(reconst2[1])
    solution = ''
    solution = get_calculations_latex(matrix_1, matrix_2, zerosum=True, bay1=data4, bay2=data5, mode=data3)
    context = solution[3]
    template = 'template.tex'
    response2 = render(request, template, context)

    if request.POST.get('pdf'):
        response = render(request, 'gaming/index.html', {'output': matrix_1,
                                                         'output2': matrix_2,
                                                         'modus': data3,
                                                         'bay1': data4,
                                                         'bay2': data5,
                                                         'sol': solution[0],
                                                         'dat': request.POST})
    return response


def latex_game(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]

    data2 = request.POST['mat2']  # Matrix Spieler 2
    reconst2 = reconstruct_matrix(data2)
    data2 = reconst2[0]
    data3 = request.POST['mod']  #
    data3 = int(data3)
    data4 = request.POST['ba1']  #
    data4 = int(data4)
    data5 = request.POST['ba2']  #
    data5 = int(data5)
    data6 = str(request.POST)
    data7 = ''
    new_game2 = Game
    matrix_1 = np.asarray(reconst1[1])
    matrix_2 = np.asarray(reconst2[1])
    solution = ''
    #solution = get_calculations_latex(matrix_1, matrix_2, zerosum=True, bay1=data4, bay2=data5, mode=data3)
    #context = solution[3]
    template2 = 'template2.tex'
    context = {'bay1': data4,
               'bay2': data5,
               'mixed': data3,
               'gamematrix': matrix_1}
    #response2 = render(request, template2, context)
    if request.POST.get('tex2'):
        response = HttpResponse(render(request, template2, context), content_type='application/x-tex')
        response['Content-Disposition'] = 'attachment; filename="aufgabe.tex"'
    return response


def latex_solution(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]
    data2 = request.POST['mat2']  # Matrix Spieler 2
    reconst2 = reconstruct_matrix(data2)
    data2 = reconst2[0]
    data3 = request.POST['mod']  #
    data3 = int(data3)
    data4 = request.POST['ba1']  #
    data4 = int(data4)
    data5 = request.POST['ba2']  #
    data5 = int(data5)
    data6 = str(request.POST)
    data7 = ''
    new_game2 = Game
    matrix_1 = np.asarray(reconst1[1])
    matrix_2 = np.asarray(reconst2[1])
    solution = ''
    solution = get_calculations_latex(matrix_1, matrix_2, zerosum=True, bay1=data4, bay2=data5, mode=data3)
    context = solution[3]
    template = 'template.tex'
    response2 = render(request, template, {'dict': context})
    if request.POST.get('tex'):
        response = HttpResponse(response2, content_type='application/x-tex')
        response['Content-Disposition'] = 'attachment; filename="solve.tex"'
    return response


def index(request):
    response = render(request, 'gaming/index2.html', {'name': 'Spiel generieren'})
    return response
