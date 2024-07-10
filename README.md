# NOTES of Machine Learning & Deep Learning

To show the hidden items:
```shell
ls -a
```
To pull and push requests by git:
```shell
git pull origin main

git status
git add .
git commit -m "messages"
git push origin <branch-name>
```
For some files you don't wanna push, then make a .gitignore file and include contents like:
```shell
file_you_dont_wanna_push.extension
```
For folders which connected a remote git repository:
```shell
cd path/to/your/folder
git init
git remote add origin https://github.com/Boris-Jobs/Kronecker_prompts.git
git add .
git commit -m "Initial messages"
git push -u origin master:main  # it means push from master to origin/main
```
Please do remember!!!
```shell
git config user.name "Boris-Jobs"
git config user.email "1322553126@qq.com"
```





## Load Pytorch on Mahti
To see which pytorch is loaded, or default, or available:
```shell
module avail pytorch
```
Or for more information:
```shell
module spider torch
module spider pytorch/2.2
```
Then, we could load the available ones and run the scripts:
```shell
module load pytorch/2.2
srun --nodes=1 --ntasks=1 --time=00:7:00 --mem-per-cpu=32G --gres=gpu:a100:2 --partition=gputest --account=project_2002243 python3 train.py
```






## Installing packages by venv
First, create a venv(virtual environment):
```shell
module load python-data
python3 -m virtualenv path/to/venv
python3 -m myvenv --system-site-packages path/to/venv
source path/to/venv/bin/activate
pip install whatshap
```
Then you could find more information by:
```shell
pip show whatshap
which python3
```
Later, when using venv:
```shell
module load python-data
source path/to/venv/bin/activate
pip install whatshap
```







## Installing packages by using pip install --user

Example to add whatshap to python-data module:
```shell
module load python-data
export PYTHONUSERBASE=path/to/venv
pip install --user whatshap
```


/scratch/project_2007023/boris/envs/missing/bin/python -m black .

/scratch/project_2007023/boris/envs/missing/bin/python -m tensorboard.main --logdir=`pwd`




## Some conda tricks
To see some config information about Python:
```shell
python -m site  # or the next conda command:
conda config --show
```
To see some information about conda:
```shell
conda info
```
To see information about envs:
```shell
echo $VIRTUAL_ENV
conda env list
```
To create a conda venv:
```shell
conda create --name PEViT python=3.12.3
```
To open command palette:
```
Ctrl + Shift + P
```
To activate my own conda env:
```shell
source /scratch/project_2007023/boris/miniconda3/bin/activate
conda activate missing
```
To run brief command by specific env:
```shell
conda run --name envname python3 -c "import sys; print(sys.executable)"
```
How to install packages by specific ways:
```shell
/scratch/project_2004030/boris/missing/bin/pip install --target=/scratch/project_2004030/boris/missing/lib/python3.8/site-packages/ -r requirements.txt

/scratch/project_2004030/boris/missing/bin/pip install --target=/scratch/project_2004030/boris/missing/lib/python3.8/site-packages/ GitPython
```








## Some tricks
显示当前目录所有文件的disk usage并按大小排序:
```shell
du -sh .[!.]* * | sort -h
```
Disk free of filesystem:
```shell
df -h .
df -h
```
Wanna see hidden file usage:
```shell
du -sh .[!.]* *  # or command like:
ls -ad .*/
```
如果根据时间来设置了result输出，要批量进行删除:
```shell
rm *_202505*
```
slurm查看GPU使用情况:
```shell
# 方法1
sinfo -o "%10N %10G %10c %10D %10T %20F" --Node | grep gpu

# 方法2
squeue -u username; ssh [nodelist]; nvidia-smi
```

查看任务:
```shell
sjstat
```
To cancel tasks:
```shell
scancel ID
```
To see details of a task:
```shell
scontrol show job JOBID
```

遇到SSH无法提交:
```shell
eval `ssh-agent -s`

chmod 600 ./.ssh/379_rsa

ssh-add ./.ssh/379_rsa

ssh -T git@github.com
```