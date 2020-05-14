def save_dict(dict_data, path):
    '''
    save dictionary to path
    '''
    with open(path, 'w', encoding = 'utf-8') as file:
        for key, value in dict_data.items():
            file.write('{}\t{}\n'.format(key, value))


def load_dict(path, int_key = True):
    '''
    load dictionary data from path
    '''
    with open(path, 'r', encoding = 'utf-8') as file:
        lines = [line.strip().split('\t') for line in file.readlines()]
        if int_key:
            dict_data = {int(line[0]): line[1] for line in lines}
        else:
            dict_data = {line[0]: int(line[1]) for line in lines}
    return  dict_data
