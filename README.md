# Kill the running script first (Ctrl+C in the other terminal)

# Remove WAL files
rm /scratch/ro31337/pubmed_index.db-shm /scratch/ro31337/pubmed_index.db-wal

# Check the DB is still usable
python -c "import sqlite3; c=sqlite3.connect('/scratch/ro31337/pubmed_index.db'); print(c.execute('SELECT COUNT(*) FROM articles').fetchone()[0], 'articles')"






find ~/anaconda3/envs/ctags -name "ctags" -type f 2>/dev/null || find ~/miniconda3/envs/ctags -name "ctags" -type f 2>/dev/null || find /opt/conda/envs/ctags -name "ctags" -type f 2>/dev/null

conda run -n ctags ctags --version | head -1


conda create -n ctags universal-ctags -c conda-forge -y

ln -s $(conda run -n ctags which ctags) ~/.local/bin/uctags

uctags --version | head -1
