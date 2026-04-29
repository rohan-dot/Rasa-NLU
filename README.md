conda create -n ctags universal-ctags -c conda-forge -y

ln -s $(conda run -n ctags which ctags) ~/.local/bin/uctags

uctags --version | head -1
