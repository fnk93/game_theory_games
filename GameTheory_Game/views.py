from gaming.Game import Game
from gaming.Solving_Methods import *
# from copy import deepcopy
from django.http import HttpResponse
from django.shortcuts import render
from gaming.postDataProcessing import reconstruct_matrix
from gaming.ZerosumGame import ZerosumGame


def game(request):
    if request.GET.get('generate'):
        new_game = ZerosumGame()
        output2 = int(request.GET.get('spielart'))
        if int(request.GET.get('spielart')) == 0 or (int(request.GET.get('spielart')) >= 2 and int(request.GET.get('spielart')) <= 5):
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
        fixed_strat_1 = np.random.randint(0, new_game.matrix.shape[0])
        fixed_strat_2 = np.random.randint(0, new_game.matrix.shape[1])
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
        response = render(request, 'gaming/index.html', {'output': matrix1,
                                                         'output2': matrix2,
                                                         'game': new_game,
                                                         'modus': mode,
                                                         'work': aufgaben_pure,
                                                         'work_mixed': aufgaben_mixed,
                                                         'bay1': fixed_strat_2,
                                                         'bay2': fixed_strat_1,})
    return response


def solution(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    # data = list(data)
    #
    # test = "".join(data)
    # test = test.replace('.', '')
    # test = test.replace('[', '')
    # test = test.split(' ')
    # new_test = []
    # for stra in test:
    #     new_test.append(stra.replace("\r\n", ''))
    # new_arr = []
    # temp = []
    # firstDigit = False
    # endFound = False
    # for i in range(len(new_test)):
    #     if "]" in new_test[i]:
    #         new_test[i] = new_test[i].replace("]", "")
    #         endFound = True
    #     if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
    #         firstDigit = True
    #         temp.append(int(new_test[i]))
    #     if (new_test[i] == "[" and temp and firstDigit) or endFound:
    #         new_arr.append(deepcopy(temp))
    #         firstDigit = False
    #         endFound = False
    #         temp.clear()
    #
    # data = deepcopy(new_arr)
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]

    data2 = request.POST['mat2']  # Matrix Spieler 2
    # data2 = list(data2)
    #
    # test2 = "".join(data2)
    # test2 = test2.replace('.', '')
    # test2 = test2.replace('[', '')
    # test2 = test2.split(' ')
    # new_test2 = []
    # for str2 in test2:
    #     new_test2.append(str2.replace("\r\n", ''))
    #
    # firstDigit = False
    # endFound = False
    # new_arr2 = []
    # temp2 = []
    # for j in range(len(new_test2)):
    #     if "]" in new_test2[j]:
    #         endFound = True
    #         new_test2[j] = new_test2[j].replace("]", "")
    #     if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
    #         firstDigit = True
    #         temp2.append(int(new_test2[j]))
    #     if (new_test2[j] == "[" and temp and firstDigit) or endFound:
    #         new_arr2.append(deepcopy(temp2))
    #         temp2.clear()
    #         firstDigit = False
    #         endFound = False
    #
    # data2 = deepcopy(new_arr2)
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
    solution = get_calculations_latex(matrix_1, matrix_2, True, data4, data5, data3)
    context = solution[2]
    template = 'template.tex'
    response2 = render(request, template, context)


def pdf_game(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    # data = list(data)
    #
    # test = "".join(data)
    # test = test.replace('.', '')
    # test = test.replace('[', '')
    # test = test.split(' ')
    # new_test = []
    # for stra in test:
    #     new_test.append(stra.replace("\r\n", ''))
    # new_arr = []
    # temp = []
    # firstDigit = False
    # endFound = False
    # for i in range(len(new_test)):
    #     if "]" in new_test[i]:
    #         new_test[i] = new_test[i].replace("]", "")
    #         endFound = True
    #     if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
    #         firstDigit = True
    #         temp.append(int(new_test[i]))
    #     if (new_test[i] == "[" and temp and firstDigit) or endFound:
    #         new_arr.append(deepcopy(temp))
    #         firstDigit = False
    #         endFound = False
    #         temp.clear()
    #
    # data = deepcopy(new_arr)
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]
    data2 = request.POST['mat2']  # Matrix Spieler 2
    # data2 = list(data2)
    #
    # test2 = "".join(data2)
    # test2 = test2.replace('.', '')
    # test2 = test2.replace('[', '')
    # test2 = test2.split(' ')
    # new_test2 = []
    # for str2 in test2:
    #     new_test2.append(str2.replace("\r\n", ''))
    #
    # firstDigit = False
    # endFound = False
    # new_arr2 = []
    # temp2 = []
    # for j in range(len(new_test2)):
    #     if "]" in new_test2[j]:
    #         endFound = True
    #         new_test2[j] = new_test2[j].replace("]", "")
    #     if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
    #         firstDigit = True
    #         temp2.append(int(new_test2[j]))
    #     if (new_test2[j] == "[" and temp and firstDigit) or endFound:
    #         new_arr2.append(deepcopy(temp2))
    #         temp2.clear()
    #         firstDigit = False
    #         endFound = False
    #
    # data2 = deepcopy(new_arr2)
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
    solution = get_calculations_latex(matrix_1, matrix_2, True, data4, data5, data3)
    context = solution[2]
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
    # data = list(data)
    #
    # test = "".join(data)
    # test = test.replace('.', '')
    # test = test.replace('[', '')
    # test = test.split(' ')
    # new_test = []
    # for stra in test:
    #     new_test.append(stra.replace("\r\n", ''))
    # new_arr = []
    # temp = []
    # firstDigit = False
    # endFound = False
    # for i in range(len(new_test)):
    #     if "]" in new_test[i]:
    #         new_test[i] = new_test[i].replace("]", "")
    #         endFound = True
    #     if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
    #         firstDigit = True
    #         temp.append(int(new_test[i]))
    #     if (new_test[i] == "[" and temp and firstDigit) or endFound:
    #         new_arr.append(deepcopy(temp))
    #         firstDigit = False
    #         endFound = False
    #         temp.clear()
    #
    # data = deepcopy(new_arr)
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]
    data2 = request.POST['mat2']  # Matrix Spieler 2
    # data2 = list(data2)
    #
    # test2 = "".join(data2)
    # test2 = test2.replace('.', '')
    # test2 = test2.replace('[', '')
    # test2 = test2.split(' ')
    # new_test2 = []
    # for str2 in test2:
    #     new_test2.append(str2.replace("\r\n", ''))
    #
    # firstDigit = False
    # endFound = False
    # new_arr2 = []
    # temp2 = []
    # for j in range(len(new_test2)):
    #     if "]" in new_test2[j]:
    #         endFound = True
    #         new_test2[j] = new_test2[j].replace("]", "")
    #     if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
    #         firstDigit = True
    #         temp2.append(int(new_test2[j]))
    #     if (new_test2[j] == "[" and temp and firstDigit) or endFound:
    #         new_arr2.append(deepcopy(temp2))
    #         temp2.clear()
    #         firstDigit = False
    #         endFound = False
    #
    # data2 = deepcopy(new_arr2)
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
    solution = get_calculations_latex(matrix_1, matrix_2, True, data4, data5, data3)
    context = solution[2]
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
    # data = list(data)
    #
    # test = "".join(data)
    # test = test.replace('.', '')
    # test = test.replace('[', '')
    # test = test.split(' ')
    # new_test = []
    # for stra in test:
    #     new_test.append(stra.replace("\r\n", ''))
    # new_arr = []
    # temp = []
    # firstDigit = False
    # endFound = False
    # for i in range(len(new_test)):
    #     if "]" in new_test[i]:
    #         new_test[i] = new_test[i].replace("]", "")
    #         endFound = True
    #     if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
    #         firstDigit = True
    #         temp.append(int(new_test[i]))
    #     if (new_test[i] == "[" and temp and firstDigit) or endFound:
    #         new_arr.append(deepcopy(temp))
    #         firstDigit = False
    #         endFound = False
    #         temp.clear()
    #
    # data = deepcopy(new_arr)
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]

    data2 = request.POST['mat2']  # Matrix Spieler 2
    # data2 = list(data2)
    #
    # test2 = "".join(data2)
    # test2 = test2.replace('.', '')
    # test2 = test2.replace('[', '')
    # test2 = test2.split(' ')
    # new_test2 = []
    # for str2 in test2:
    #     new_test2.append(str2.replace("\r\n", ''))
    #
    # firstDigit = False
    # endFound = False
    # new_arr2 = []
    # temp2 = []
    # for j in range(len(new_test2)):
    #     if "]" in new_test2[j]:
    #         endFound = True
    #         new_test2[j] = new_test2[j].replace("]", "")
    #     if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
    #         firstDigit = True
    #         temp2.append(int(new_test2[j]))
    #     if (new_test2[j] == "[" and temp and firstDigit) or endFound:
    #         new_arr2.append(deepcopy(temp2))
    #         temp2.clear()
    #         firstDigit = False
    #         endFound = False
    #
    # data2 = deepcopy(new_arr2)
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
    solution = get_calculations_latex(matrix_1, matrix_2, True, data4, data5, data3)
    context = solution[2]
    template = 'template.tex'
    response2 = render(request, template, context)
    if request.POST.get('tex2'):
        template2 = 'template2.tex'
        if context['solvemixed'] != "":
            context[
                'mixed'] = r'Und folgendes soll für gemischte Strategien ermittelt werden:\\ \begin{itemize} \item Optimale Strategienkombination der beiden Spieler inklusive Auszahlung \end{itemize}'
        else:
            context['mixed'] = ''
        response = HttpResponse(render(request, template2, context), content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename="aufgabe.tex"'
    return response


def latex_solution(request):
    data = request.POST['mat1']  # Matrix Spieler 1
    # data = list(data)
    #
    # test = "".join(data)
    # test = test.replace('.', '')
    # test = test.replace('[', '')
    # test = test.split(' ')
    # new_test = []
    # for stra in test:
    #     new_test.append(stra.replace("\r\n", ''))
    # new_arr = []
    # temp = []
    # firstDigit = False
    # endFound = False
    # for i in range(len(new_test)):
    #     if "]" in new_test[i]:
    #         new_test[i] = new_test[i].replace("]", "")
    #         endFound = True
    #     if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
    #         firstDigit = True
    #         temp.append(int(new_test[i]))
    #     if (new_test[i] == "[" and temp and firstDigit) or endFound:
    #         new_arr.append(deepcopy(temp))
    #         firstDigit = False
    #         endFound = False
    #         temp.clear()
    #
    # data = deepcopy(new_arr)
    reconst1 = reconstruct_matrix(data)
    data = reconst1[0]
    data2 = request.POST['mat2']  # Matrix Spieler 2
    # data2 = list(data2)
    #
    # test2 = "".join(data2)
    # test2 = test2.replace('.', '')
    # test2 = test2.replace('[', '')
    # test2 = test2.split(' ')
    # new_test2 = []
    # for str2 in test2:
    #     new_test2.append(str2.replace("\r\n", ''))
    #
    # firstDigit = False
    # endFound = False
    # new_arr2 = []
    # temp2 = []
    # for j in range(len(new_test2)):
    #     if "]" in new_test2[j]:
    #         endFound = True
    #         new_test2[j] = new_test2[j].replace("]", "")
    #     if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
    #         firstDigit = True
    #         temp2.append(int(new_test2[j]))
    #     if (new_test2[j] == "[" and temp and firstDigit) or endFound:
    #         new_arr2.append(deepcopy(temp2))
    #         temp2.clear()
    #         firstDigit = False
    #         endFound = False
    #
    # data2 = deepcopy(new_arr2)
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
    solution = get_calculations_latex(matrix_1, matrix_2, True, data4, data5, data3)
    context = solution[2]
    template = 'template.tex'
    response2 = render(request, template, context)
    if request.POST.get('tex'):
        response = HttpResponse(response2, content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename="solve.tex"'
    return response


def index(request):
    matrix1 = ''
    matrix2 = ''
    new_game = ''
    mode = ''
    aufgaben_pure = ''
    aufgaben_mixed = ''
    fixed_strat_2 = ''
    fixed_strat_1 = ''
    response = render(request, 'gaming/index.html', {})
    # mode = 0 reine Strategien
    # mode = 1 gemischte Strategien
    # if request.POST.get('pdf') or request.POST.get('tex') or request.POST.get('pdf2') or request.POST.get('tex2'):
    #     data = request.POST['mat1'] # Matrix Spieler 1
    #     data = list(data)
    #
    #     test = "".join(data)
    #     test = test.replace('.', '')
    #     test = test.replace('[', '')
    #     test = test.split(' ')
    #     new_test = []
    #     for stra in test:
    #         new_test.append(stra.replace("\r\n", ''))
    #     new_arr = []
    #     temp = []
    #     firstDigit = False
    #     endFound = False
    #     for i in range(len(new_test)):
    #         if "]" in new_test[i]:
    #             new_test[i] = new_test[i].replace("]", "")
    #             endFound = True
    #         if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
    #             firstDigit = True
    #             temp.append(int(new_test[i]))
    #         if (new_test[i] == "[" and temp and firstDigit) or endFound:
    #             new_arr.append(deepcopy(temp))
    #             firstDigit = False
    #             endFound = False
    #             temp.clear()
    #
    #     data = deepcopy(new_arr)
    #
    #     data2 = request.POST['mat2'] # Matrix Spieler 2
    #     data2 = list(data2)
    #
    #     test2 = "".join(data2)
    #     test2 = test2.replace('.', '')
    #     test2 = test2.replace('[', '')
    #     test2 = test2.split(' ')
    #     new_test2 = []
    #     for str2 in test2:
    #         new_test2.append(str2.replace("\r\n", ''))
    #
    #     firstDigit = False
    #     endFound = False
    #     new_arr2 = []
    #     temp2 = []
    #     for j in range(len(new_test2)):
    #         if "]" in new_test2[j]:
    #             endFound = True
    #             new_test2[j] = new_test2[j].replace("]", "")
    #         if new_test2[j].isdigit() or (new_test2[j].startswith('-') and new_test2[j][1:].isdigit()):
    #             firstDigit = True
    #             temp2.append(int(new_test2[j]))
    #         if (new_test2[j] == "[" and temp and firstDigit) or endFound:
    #             new_arr2.append(deepcopy(temp2))
    #             temp2.clear()
    #             firstDigit = False
    #             endFound = False
    #
    #     data2 = deepcopy(new_arr2)
    #
    #     data3 = request.POST['mod'] #
    #     data3 = int(data3)
    #     data4 = request.POST['ba1'] #
    #     data4 = int(data4)
    #     data5 = request.POST['ba2'] #
    #     data5 = int(data5)
    #     data6 = str(request.POST)
    #     data7 = ''
    #     new_game2 = Game
    #     matrix_1 = np.asarray(new_arr)
    #     matrix_2 = np.asarray(new_arr2)
    #     solution = ''
    #     solution = get_calculations_latex(matrix_1, matrix_2, True, data4, data5, data3)
    #     context = solution[2]
    #     template = 'template.tex'
    #     response2 = render(request, template, context)
    #     if request.POST.get('tex'):
    #         response = HttpResponse(response2, content_type='application/force-download')
    #         response['Content-Disposition'] = 'attachment; filename="solve.tex"'
    #     if request.POST.get('pdf'):
    #         response = render(request, 'gaming/index.html', {'output': matrix_1,
    #                                                          'output2': matrix_2,
    #                                                          'modus': data3,
    #                                                          'bay1': data4,
    #                                                          'bay2': data5,
    #                                                          'sol': solution[0],
    #                                                          'dat': request.POST})
    #     if request.POST.get('tex2'):
    #         template2 = 'template2.tex'
    #         if context['solvemixed'] != "":
    #             context['mixed'] = r'Und folgendes soll für gemischte Strategien ermittelt werden:\\ \begin{itemize} \item Optimale Strategienkombination der beiden Spieler inklusive Auszahlung \end{itemize}'
    #         else:
    #             context['mixed'] = ''
    #         response = HttpResponse(render(request, template2, context), content_type='application/force-download')
    #         response['Content-Disposition'] = 'attachment; filename="aufgabe.tex"'
    #     if request.POST.get('pdf2'):
    #         aufgaben_pure = ['Maximin-Strategie(n) beider Spieler',
    #                          'Indeterminiertheitsintervall',
    #                          'Untere Spielwerte der Spieler',
    #                          'Garantiepunkt des Spiels',
    #                          'Bayes-Strategie von Spieler 1 gegenüber Strategie ' + str(
    #                              data4 + 1) + ' von Spieler 2',
    #                          'Bayes-Strategie von Spieler 2 gegenüber Strategie ' + str(
    #                              data5 + 1) + ' von Spieler 1',
    #                          'Gleichgewichtspunkt und dessen Auszahlungen']
    #         aufgaben_mixed = ['Gleichgewichtspunkt des Spiels',
    #                           'Strategienkombination im Gleichgewicht',
    #                           'Auszahlung für die Spieler']
    #         response = render(request, 'gaming/index.html', {'output': matrix_1,
    #                                                             'output2': matrix_2,
    #                                                             'modus': data3,
    #                                                             'bay1': data4,
    #                                                             'bay2': data5,
    #                                                             'work': aufgaben_pure,
    #                                                             'work_mixed': aufgaben_mixed,})

    # elif request.GET.get('generate'):
    #     new_game = Game()
    #     output2 = int(request.GET.get('spielart'))
    #     if int(request.GET.get('spielart')) == 0 or (int(request.GET.get('spielart')) >= 2 and int(request.GET.get('spielart')) <= 5):
    #         mode = 0
    #     else:
    #         mode = 1
    #     if int(request.GET.get('spielart')) == 2:
    #         new_game.matrix = np.asarray([[6,5,6,5],
    #                                       [1,4,2,-1],
    #                                       [8,5,7,5],
    #                                       [0,2,6,2]])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 3:
    #         new_game.matrix = np.asarray([[4,1,2],
    #                                       [1,5,0],
    #                                       [4,3,3],])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 4:
    #         new_game.matrix = np.asarray([[3,0,2],
    #                                       [-4,-1,-3],
    #                                       [2,-2,-1],])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 5:
    #         new_game.matrix = np.asarray([[4,5,2],
    #                                       [6,3,2],])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 6:
    #         new_game.matrix = np.asarray([[1,7,0,3],
    #                                       [0,0,3,5],
    #                                       [1,2,4,1],
    #                                       [6,0,2,0]])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 7:
    #         new_game.matrix = np.asarray([[8,6],
    #                                       [4,7],])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 8:
    #         new_game.matrix = np.asarray([[2,5],
    #                                       [4,3],
    #                                       [3,6],
    #                                       [5,4],
    #                                       [4,4]])
    #         new_game.matrix2 = new_game.matrix * -1
    #     elif int(request.GET.get('spielart')) == 9:
    #         new_game.matrix = np.asarray([[3,0,2],
    #                                       [4,5,1],
    #                                       [2,2,-1],])
    #         new_game.matrix2 = new_game.matrix * -1
    #     matrix1 = new_game.matrix
    #     matrix2 = new_game.matrix2
    #     fixed_strat_1 = randrange(0, new_game.matrix.shape[0])
    #     fixed_strat_2 = randrange(0, new_game.matrix.shape[1])
    #     aufgaben_pure = ['Maximin-Strategie(n) beider Spieler',
    #                      'Indeterminiertheitsintervall',
    #                      'Untere Spielwerte der Spieler',
    #                      'Garantiepunkt des Spiels',
    #                      'Bayes-Strategie von Spieler 1 gegenüber Strategie ' + str(fixed_strat_2 + 1) + ' von Spieler 2',
    #                      'Bayes-Strategie von Spieler 2 gegenüber Strategie ' + str(fixed_strat_1 + 1) + ' von Spieler 1',
    #                      'Gleichgewichtspunkt und dessen Auszahlungen']
    #     aufgaben_mixed = ['Gleichgewichtspunkt des Spiels',
    #                       'Strategienkombination im Gleichgewicht',
    #                       'Auszahlung für die Spieler']
    #     response = render(request, 'gaming/index.html', {'output': matrix1,
    #                                                      'output2': matrix2,
    #                                                      'game': new_game,
    #                                                      'modus': mode,
    #                                                      'work': aufgaben_pure,
    #                                                      'work_mixed': aufgaben_mixed,
    #                                                      'bay1': fixed_strat_2,
    #                                                      'bay2': fixed_strat_1,})
    return response
