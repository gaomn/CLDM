
def get_arguments(arg_list, arg_dict):
    for arg in arg_list:
        if arg in list(arg_dict.keys()):
            return arg_dict[arg]
    print(f'Missing Argument : {arg_dict[0]}')
    raise


def check_arguments(arg_list, arg_dict, default_value):
    for arg in arg_list:
        if arg in list(arg_dict.keys()):
            return arg_dict[arg]
    return default_value

