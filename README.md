find ~/anaconda3/envs/ctags -name "ctags" -type f 2>/dev/null || find ~/miniconda3/envs/ctags -name "ctags" -type f 2>/dev/null || find /opt/conda/envs/ctags -name "ctags" -type f 2>/dev/null

conda run -n ctags ctags --version | head -1


conda create -n ctags universal-ctags -c conda-forge -y

ln -s $(conda run -n ctags which ctags) ~/.local/bin/uctags

uctags --version | head -1
