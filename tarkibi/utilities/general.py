import os

BASE_DIR = '.tarkibi'

def make_directories(directories: list) -> None:
    for path in directories:
        dirs = path.split('/')
        for i in range(1, len(dirs) + 1):
            path = '/'.join(dirs[:i])
            if not os.path.exists(path):
                os.mkdir(path)
        
