# Create new env 
conda create -n nlp2 python=3.7   # There is no tensorflow for py3.8
source activate nlp2

# add new kernel 
conda install ipykernel  
python -m ipykernel install --user --name nlp2 --display-name "Python37(nlp2)"
conda install -n nlp2  nb_conda_kernels

yes | conda install -n nlp2 -c conda-forge numpy pandas tqdm sklearn scipy matplotlib seaborn



# pip installations
/Users/poudel/opt/miniconda3/envs/nlp2/bin/pip install transformers[tf-cpu]