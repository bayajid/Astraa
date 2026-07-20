
def p2f(prec):
    # function to process GPs ephemeris file data- from fortran double digit to floats
    return float(prec.replace('D','e'))
def str2lst(st):
    # function to convert string of a list into a list of int
    lst = st.split(' ') # convert to list
    lst_int = [int(num) for num in lst if num != '']
    return lst_int
def get_label_names(case_list):
    """make nice sat names

    Args:
        case_list (_type_): list of satellite names, automatically laballed

    Returns:
        case_label: cleaned up list of names
    """    # function to convert satellite names to cleaner labels
    # input lsit of sat names
    case_label = [name.replace('sat_leo_incl_', 'LEO_I_') for name in case_list]
    if case_list == case_label or 1:
        case_label = [name.replace('sat_leo_polar_', 'LEO_P_') for name in case_label]
    if case_list == case_label or 1:
        case_label = [name.replace('sat_meo_0', 'MEO') for name in case_label]
    
    case_label = [name.replace('link', 'Link ') for name in case_label]
    case_label = [name.replace('leo_incl_', 'LEO_I_') for name in case_label]
    case_label = [name.replace('leo_polar_', 'LEO_P_') for name in case_label]
    case_label = [name.replace('_meo_', ' MEO_') for name in case_label]
    case_label = [name.replace('_LEO', ' LEO') for name in case_label]
    return case_label