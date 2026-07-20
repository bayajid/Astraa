# parse string of list of floats 
def parse_col(col, delim = ','):
    col_strip = col.strip('[').strip(']')
    col_strip2 = col_strip.replace(delim,'')
    col_split = col_strip2.split(' ')
    lst = []
    for val in col_split:
        if val != '':
            lst.append(float(val))
    return lst