#/bin/python
import sys
import os

wd = os.getenv("HOME")
# Example of wd = '/home/1169'
#print("Working Directory", wd)
wd_id = wd.split('/')[-1] # This is for creating names of docker container with only user id


# quick .sh is a sample slurm file. It has to be in the same directory as this script
with open('quick.sh','r') as fo:    
    slurm_file_list = fo.readlines()    # Reading the txt file line by line as a list

# Getting the arguments passed by the user 
# For example : python quickstart.py test hello
# In that case sys.argv[1] = test and sys.argv[2] = hello

job_name = sys.argv[1]  #  No constraints - slurm job name
job_type = sys.argv[2]  # 'test' or 'run'
gpu = sys.argv[3]
python_ver = sys.argv[4]
tf_pytorch = sys.argv[5]
path_for_main = sys.argv[6] # Has to be in .py format

# Creating Slurm file based on user inputs

# giving runtime and partition based on job_type
if job_type == 'test':
    time = '00:10:00'   
    partition = 'test'
else:
    time = 'infinite'    
    partition = 'run'
  
# slurm_file_list[1] corresponds to 2nd line of quick.sh
# rstrip us being used to remove the orgiinal \n 
slurm_file_list[1] = slurm_file_list[1].rstrip() + job_name + '\n'
slurm_file_list[3] = slurm_file_list[3].rstrip() + time + '\n'
slurm_file_list[7] = slurm_file_list[7].rstrip() + partition + '\n'
if (int(gpu) > 2):
    slurm_file_list[5] = slurm_file_list[5].rstrip() + str(2) + '\n'
else:
    slurm_file_list[5] = slurm_file_list[5].rstrip() + str(gpu) + '\n'


# For the last line, we first create a list of words that will be written in the line
# slurm_file_list[-1] means the last element of the list
# split is basically dividing the line into a list of words seperated by space ' '
words_list = slurm_file_list[-1].split(' ')
#print("Before", words_list)
#print(job_type)
if python_ver == "python2":
    words_list[8] = 'python2'
else:
    words_list[8] = 'python3'
if tf_pytorch == 'tf':
    if len(sys.argv)== 8:
        slurm_file_list.insert(9, 'docker build -t ' +wd_id + '-tensorflow' + ' .' +  '\n')
        words_list[7] = wd_id + '-tensorflow'
    else:
        words_list[7] = 'quickstart' + '-tensorflow'
else:
    if len(sys.argv)== 8:
        # docker build command only inserted if requirements.txt is passed
        slurm_file_list.insert(9, 'docker build -t ' +wd_id + '-pytorch' + ' .' +'\n')
        # Container name will be given as userid-tf otherwise it will be given quickstart-tf
        words_list[7] = wd_id + '-pytorch'
    else:
        words_list[7] = 'quickstart' + '-pytorch'
# the following two do not depend on tf/pyt or requirements.txt
words_list[4] = str(wd) + ':/home'
words_list[-3] = path_for_main

# Now adding the words_list as last line in slurm_file_list
slurm_file_list[-1] = " ".join(words_list)
#print("After", words_list)

# Creating a SLURM file
with open('user_slurm.sh', 'w') as fw:
    for listitem in slurm_file_list:
        fw.write('%s' % listitem)


# SLURM file has been created and saved

# Now creating docker file IF user has passed requirements.txt

if len(sys.argv)== 8:
    print("heello")
    path_for_requirements = sys.argv[7]
    with open('user_Dockerfile','r') as fd:
        dockerfile_list = fd.readlines()
    if tf_pytorch == 'pyt':
        dockerfile_list[0] = 'FROM nvcr.io/nvidia/pytorch:19.08-py3' + '\n'
    else:
        dockerfile_list[0] = 'FROM nvcr.io/nvidia/tensorflow:19.05-py3' + '\n'
    dockerfile_list[2] = 'COPY ' + path_for_requirements + ' .'+ '\n'
    dockerfile_list[3] = 'RUN pip install -r ' + path_for_requirements

    with open('Dockerfile', 'w') as fdw:
        print("file created")
        for listitem in dockerfile_list:
            fdw.write('%s' % listitem)
    #if tf_pytorch == 'tf':
        #os.system('docker build -t $USER-tensorflow .')
    #else :
        #os.system('docker build -t $USER-pytorch .')

os.system('sbatch user_slurm.sh')
