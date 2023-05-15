# basketball_registration
#  SO-SLAM


<p align="center">
  <img src="https://github.com/MRHan-426/SOSLAM/blob/master/.assets/3%2000_00_00-00_00_30.gif" alt="gif">
</p>
     
![build passing](https://img.shields.io/badge/build-passing-brightgreen)
[![License](https://img.shields.io/github/license/MRHan-426/SOSLAM)](./LICENSE.txt)
![Primary language](https://img.shields.io/github/languages/top/MRHan-426/SOSLAM)
![ROB530](https://img.shields.io/badge/ROB530-group6-orange)


This is Team 6's final project git repository for ROB530: Mobile Robotics. 

The title of our project is **Implementation and Evaluation of Semantic-Object SLAM Algorithm**

The team members include: Ziqi Han, Zhewei Ye, Tien-Li Lin, Yi-Cheng Liu, Shubh Agrawal.

**Related Paper:**  [RA-L&ICRA 2022]

+ Liao Z, Hu Y, Zhang J, et al. So-slam: Semantic object slam with scale proportional and symmetrical texture constraints[J]. IEEE Robotics and Automation Letters, 2022, 7(2): 4008-4015. [**[PDF]**](https://arxiv.org/abs/2109.04884)

---

## 📚 1. Prerequisites


```shell
sudo apt-get install libglew-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libtbb-dev
sudo apt-get install libmetis-dev
sudo apt-get install libpugixml-dev
sudo apt-get install libpcl-dev
```


```shell
cmake 3.26.0
libboost 1.71.0  # make sure to compile C++ version from source code.
Pangolin 0.8.0
OpenCV 4.7.0
```



## ⚙️ 2. Compile GTSAM

**Note that higher version may bring unexpected errors, we do not test other version so far.**

```shell
git clone --branch 4.1.1 https://github.com/borglab/gtsam.git
```

Modify Eigen cmake config file: cmake/HandleEigen.cmake

```shell
set(GTSAM_USE_SYSTEM_EIGEN ON)
```

Then:

```shell
mkdir build && cd build
cmake ..
make check
sudo make install
```




## 🛠️ 3. Compile our repo

Branch Master contains point cloud visualization, so you have some more prerequisites.

```shell
git clone --branch master https://github.com/MRHan-426/SOSLAM.git
```

Branch 0.0.1 doesnot contain point cloud visualization, so you don't have to compile PCL, VTK.

```shell
git clone --branch 0.0.1 https://github.com/MRHan-426/SOSLAM.git
```

Then:

```shell
mkdir build
cmake ..
make
```




## 🌟 4. Examples





## 🎬 5. Videos and Documentation

+ Our project presentation video is on [**[YouTube]**](https://youtu.be/_yUy5nOtfMM).




+ Project Document: [**[PDF]**](TODO)




## 📝 6. Note

+ If you want to use it in your work or with other datasets, you should prepare the dataset containing:

  - RGB image
  - Label xml (contain "objectKey" key to store the data association information)
  - Odom txt
  - Depth image (if you do not need point cloud visualization, just ignore)
  - Camera intrinsic txt

  Be aware that you should rename your images and xmls as number 1,2,3,...

  Be aware that RGB, Depth, Label, Odom must match.

+ This is an incomplete version of our project. 
    - We have a lot of experiments to be done.
    - We have not achieved real-time.




## 🏅 7. Acknowledgement

Thanks for the great work: 

+ [**SO-SLAM**](https://github.com/XunshanMan/SoSLAM)
+ [**GTSAM**](https://github.com/borglab/gtsam)
+ [**Quadric-SLAM**](https://github.com/qcr/quadricslam)
+ [**EAO-SLAM**](https://github.com/yanmin-wu/EAO-SLAM) 
+ [**ORB-SLAM2**](https://github.com/raulmur/ORB_SLAM2)
+ [**YOLO-v8**](https://github.com/ultralytics/ultralytics)



## 📫 8. Contact

+ Tien-Li Lin, Email: tienli@umich.edu
+ Yi-Cheng Liu, Email: liuyiche@umich.edu





**Please cite the author's paper if you use the code in your work.**

```
@article{liao2022so,
  title={So-slam: Semantic object slam with scale proportional and symmetrical texture constraints},
  author={Liao, Ziwei and Hu, Yutong and Zhang, Jiadong and Qi, Xianyu and Zhang, Xiaoyu and Wang, Wei},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={4008--4015},
  year={2022},
  publisher={IEEE}
}
```

