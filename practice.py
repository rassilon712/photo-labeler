import os
import subprocess
dir_path = os.path.dirname(os.path.realpath(__file__))

os.system("start cmd.exe @cmd @echo off /c python practice1.py")
os.system("start cmd.exe @cmd @echo off /c python practice2.py")
os.system("start cmd.exe @cmd @echo off /c python practice3.py")

# subprocess.call("python practice1.py")
# subprocess.call("python practice2.py")
# subprocess.call("python practice3.py")
# os.system("start python practice1.py")
# print(type(new))
# print(new)
# print(dir_path)
# os.system('python practice1.py')
