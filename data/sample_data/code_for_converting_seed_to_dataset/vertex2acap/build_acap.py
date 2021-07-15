import os

import argparse

parser = argparse.ArgumentParser(description='Distributing all data')
parser.add_argument('-r', type=int, default=0, metavar='N',
                    help='run (default: 0)')
args = parser.parse_args()
if_run = args.r

if os.path.isdir("./build"):
    pass
else:
    os.mkdir("./build")
os.chdir("./build")
os.system("rm -rf ./*")
os.system("cmake -DCMAKE_BUILD_TYPE=Release ..")
os.system("make")

if if_run:
    os.chdir("..")
    f=open("./CMakeLists.txt")
    line=f.readline()
    w=line.split(' ')
    os.system("./build/"+w[1])
