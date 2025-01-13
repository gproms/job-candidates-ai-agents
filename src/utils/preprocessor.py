def extract_name_from_file_name(file_name):
    """
    Extracts a candidate's name from file names like `cv_<name>.txt` or `interview_<name>.txt`.
    """
    return file_name.split('_')[1].split('.')[0].title()
