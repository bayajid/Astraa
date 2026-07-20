import json

def dict2txt(dict, name, folder, print_cond = 1):
    """Saving dictionaries into JSON files

    Args:
        dict (dict): Dictionary to be saved
        name (str): name of file to be saved (w/o file extension)
        folder (path): complete or relative path to save the file
        print_cond (int, optional): Conditional to print whether saves were successfull. Defaults to 1.
    """    
    if print_cond:
        print(f'Attempting to save {name}.')
    try:
        with open(f'{folder}/{name}.json', "w") as json_to_save:
            json.dump(dict, json_to_save, indent = 4)
        if print_cond:
            print(f'{name} saved succesfully.')
    except Exception as e:
        if print_cond:
            print(e)
            print(f'Failed to save {name}.')
            
if __name__ == "__main__":
    a = {'a' : 1}
    path = 'temp'
    dict2txt(a, 'a', path)