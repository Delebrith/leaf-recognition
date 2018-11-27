import os


def get_files_in_classes(data_src):
    class_directory = os.listdir(data_src)[0]
    directories = []
    classes = []
    for element in os.listdir(os.path.join(data_src, class_directory)):
        path = os.path.join(os.path.join(data_src, class_directory), element)
        if os.path.isdir(path):
            directories.append(path)
            classes.append(element)

    images = []
    for directory in directories:
        for element in os.listdir(directory):
            path = os.path.join(directory, element)
            if os.path.isfile(path):
                images.append(path)

    return images, classes
