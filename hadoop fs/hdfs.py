# import the python subprocess module
import subprocess
from logging import exception
import os
path = "/"
oldpath = ""
host="localhost"
port="9870"



def run_cmd(args_list):
        """
        run cmd commands
        """
        # a=os.popen(' '.join(args_list),"r")
        result = []
        process = subprocess.Popen(f"%hadoop_home%/bin/hdfs dfs", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        b=process.stdout
        for line in process.stdout:
            result.append(line)
        for line in result:
            print(line)
        pass

def Path_back(s):
    if s.count("/")==1:
        return -1
    if s[-1]=="/":
        s=s[:-1]
    rpath = s[::-1]
    idx = len(s) - rpath.index("/") - 1
    return s[0:idx]

def checkFilePath(path:str):
    if not os.path.exists(path):
        return 0
    if os.path.isdir(str):
        return 2
    return 1
    
def checkHDFSpath(path:str):
    toCheck = run_cmd(["hdfs","dfs","-ls",path])
    if("No such file or directory" in toCheck):
        return False
    return True


def put_hdfs(file: str,path: str,isfolder: bool=False):
    checkFile = checkFilePath(file)
    checkPath = checkHDFSpath(path)
    errors = ""
    if not checkFile:
        errors+="File doesn't exist on local system.\n"
    if not checkPath:
        errors+="Path doesn't exist on HDFS.\n"
    if checkFile==2 and not isfolder:
        errors+="Path is a folder yet folder not indicated.\n"
    if errors!="":
        raise exception(errors)
        
    
put_hdfs("","")


# while(True):
#     args=["hdfs","dfs","-ls",path]
#     a=run_cmd(args)
#     if("No such file or directory" in a or a==''):
#         print("no files/folder found with that name, try again")
#         pass
#     else:
#         print(a)
#         oldpath=path
#     path=input("Input path:")
    
# print(Path_back("/one/two/three/"))
