
def save_dataset_as_csv(dataset, path_name):
    '''
    <font color="#f56e62">perform clustering over the embeddings.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **path_name** : <font color="#008001">str</font>
        <br>path where the file needs to be saved.
    '''
    dataset.to_csv(path_name, index=False)
