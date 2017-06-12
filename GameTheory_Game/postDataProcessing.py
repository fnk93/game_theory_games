from copy import deepcopy


def reconstruct_matrix(post_data):
    data = list(post_data)

    test = "".join(data)
    test = test.replace('.', '')
    test = test.replace('[', '')
    test = test.split(' ')
    new_test = []
    for stra in test:
        new_test.append(stra.replace("\r\n", ''))
    new_arr = []
    temp = []
    first_digit = False
    end_found = False
    for i in range(len(new_test)):
        if "]" in new_test[i]:
            new_test[i] = new_test[i].replace("]", "")
            end_found = True
        if new_test[i].isdigit() or (new_test[i].startswith('-') and new_test[i][1:].isdigit()):
            first_digit = True
            temp.append(int(new_test[i]))
        if (new_test[i] == "[" and temp and first_digit) or end_found:
            new_arr.append(deepcopy(temp))
            first_digit = False
            end_found = False
            del temp[:]

    data = deepcopy(new_arr)
    return data, new_arr
