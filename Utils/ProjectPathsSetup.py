import sys
import os

class ProjectPathsSetup():
    
    def __init__(self):
        
        pass
        
    def __get_all_directories(self, directory):
        directories = []
        for root, _, _ in os.walk(directory):
            directories.append(root)
        return directories

 
        

    def add_project_paths(self, project_path):
        
        all_directories = self.__get_all_directories(project_path)
        for dir in all_directories:
            sys.path.append(dir)
            
            
   