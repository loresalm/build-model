

def log(txt: str, verbose: bool):
    '''
    <font color="#f56e62">format and print the text if verbose is true.</font>

    Parameters
    ===============

    **txt** : <font color="#008001">str</font>
        <br>The text that needs to be printed

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)
    '''
    if verbose:
        print(f'----- {txt} -----')
