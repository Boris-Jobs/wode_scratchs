# wode_scratchs


I first realized the significance of "scratch" during the Harvard's ['CS50'](https://cs50.harvard.edu/x/2024/) course, where they built many foundational concepts of computing from scratch.   

Inspired by that, I've decided to create this repository to document the four topics I am passionate about, __machine learning algorithms, deep learning frameworks, large language models, and data structures and algorithms__ — all built from scratch. You'll find these files under the `./scratchs/` directory.

The following section, `boris's Common Command Manual`, contains some useful commands I frequently use when working with `git` and clusters. Feel free to explore if you're interested.




## boris's Common Command Manual


To show the hidden items:
```shell
ls -a
```
To pull and push requests by git:
```shell
git pull

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
git remote add origin https://github.com/Boris-Jobs/xxx.git
git add .
git commit -m "Initial messages"
git push -u origin master:main  # it means push from master to origin/main
```
Please do remember!!!
```shell
git config user.name "Boris-Jobs"
git config user.email "13xxx126@qq.com"
```
Using SSH:
```shell
ssh-keygen -t ed25519 -C "13xxx126@qq.com"
# ed25519 意味着是基于 Ed25519 椭圆曲线算法生成密钥对

eval `ssh-agent -s`

chmod 600 /scratch/project_20xxx8/v/.ssh/your_ssh_file

ssh-add /scratch/project_20xxx8/v/.ssh/your_ssh_file

cat /scratch/project_20xxx8/v/.ssh/your_ssh_file.pub  
# 添加内容到GitHub的ssh key里

git remote -v

ssh -T git@github.com
```




## Load Pytorch on Mahti
可参考:
[https://docs.csc.fi/apps/pytorch/](https://docs.csc.fi/apps/pytorch/)  

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
srun --nodes=1 --ntasks=1 --time=00:7:00 --mem-per-cpu=32G --gres=gpu:a100:2 --partition=gputest --account=project_2xxxx2 python3 train.py
```






## Installing packages by venv in Mahti
First, create a venv(virtual environment):
```shell
module load python-data
python3 -m venv --system-site-packages path/to/venv
# python3 -m venv --system-site-packages 是固定的
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
可参考:
[https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules](https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules)

Example to add whatshap to python-data module:
```shell
module load python-data
export PYTHONUSERBASE=path/to/venv
pip install --user whatshap
```

## Tricks
美化代码
/scratch/project_xxx23/boris/envs/missing/bin/python -m black .

/scratch/project_xxx23/boris/envs/missing/bin/python -m tensorboard.main --logdir=`pwd`




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
source /scratch/project_20xxxx3/boris/miniconda3/bin/activate
conda activate your_venv
```
To run brief command by specific env:
```shell
conda run --name envname python3 -c "import sys; print(sys.executable)"
```
How to install packages by specific ways:
```shell
/scratch/project_20xxxx0/xxx/missing/bin/pip install --target=/scratch/project_xxxx/xx/lib/python3.8/site-packages/ -r requirements.txt

/scratch/project_2xxx030/boris/missing/bin/pip install --target=/scratch/project_xxx/xx/lib/python3.8/site-packages/ GitPython
```








## Other tricks
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


