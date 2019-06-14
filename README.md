# NIID-MSL
This is the source code of the paper ["Robust Multi-view Subspace Learning with Non-independent
and Non-identically Distributed Complex Noise"](http://gr.xjtu.edu.cn/c/document_library/get_file?folderId=2618163&name=DLFE-118022.pdf).
We provided two implementation of the proposed NIID-MSL in the floder ./varaiatoin, hdp_multi_view.m and
hdp_multi_view_cell.m. The former one is followed the regular VB only using MATLAB and latter
one is accelerated through the strategies of the Section V.C and mix-programing with Matlab
and C. If you have any questions, please feel free to contact me (zsy20110806207@stu.xjtu.edu.cn).

# Experimental Environment
* MATLAB 2018a
* Ubuntu 16.04
* C Compiler: Gcc 6.3

# How to run
1. Download the dataset from the following link and put them in the ./data floder.
*Google Drive link: https://drive.google.com/drive/folders/12aFroF3aR7t5PwmlLt4WGiAW33XpD9E0?usp=sharing
*Baidu Cloud link: https://pan.baidu.com/s/1_-wFJGrTARPd_6iRkJhYgg, Password: 7ila

2. Switch to the root path of thei project and add path to matlab
<pre><pre><code> init_path </code></pre></pre>
2. Run any of the demo listed in the ./demo floder, such as
<pre><pre><code> demo_scurve </code></pre></pre>


