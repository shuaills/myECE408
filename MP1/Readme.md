

## Lab1(MP1) Instructions

If you have already checked out the repo before 1/23 12AM, you need to rerun `git pull` to make sure everything is up to date.

To submit your code, you need to run `rai -p ./MP1 --submit MP1`. To check your submission history, run `rai history -p ./MP1` (last 20 enties) or `rai l-history -p ./MP1` (last 100 enties). 

You will need the URL to your submission to complete the first question of your lab1 quiz. The quiz is autograded, and the URL is saved for regrading purpose. Make sure to answer each question before turning in the quiz. The lab and quiz due Feb 3 at 8pm. 


### Objective

The purpose of this lab is for you to become familiar with using the CUDA API by implementing a simple vector addition kernel and its associated host code as shown in the lectures.

### Prerequisites

Before starting this lab, make sure that:

* You have completed all week 1 lectures or videos

* You have completed Lab0 (MP0)

* Chapter 2 of the text book would also be helpful

### Instruction

You should edit the code in `template.cu` to perform the following:

* Allocate device memory

* Copy host memory to device

* Initialize thread block and kernel grid dimensions

* Invoke CUDA kernel

* Copy results from device to host

* Free device memory

* Write the CUDA kernel

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.

You can test your code by running `rai -p ./MP1`. If your solution is 
correct, you should be able to see the following output for each of 
the 10 test datasets:
```
--------------
Dataset  X
The input length is XX
Solution is correct
```

### WSL issue with run_datasets

Windows may convert newline character of our run_datasets script, which will cause execution failure (executing command $'\r') on the RAI server. To fix this, run the following commands to convert the script back to Unix:
```
sudo apt-get install dos2unix
dos2unix run_datasets
```
