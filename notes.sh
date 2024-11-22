wget http://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -zxvf Python-3.10.14.tgz
cd Python-3.10.14
mkdir ~/.localpython
./configure --prefix=/home/nk9431/.localpython
make
make install
PATH=$PATH:~/.localpython/bin

# add PYTHONPATH to env variables in windows
# close and reopen vscode
# in ipynb {
#   import os
#   print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
# }
# open new terminal window
echo $env:PYTHONPATH