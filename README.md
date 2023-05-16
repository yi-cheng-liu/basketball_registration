# Player Tracking in Bird-Eye Diagram Through Basketball Video
     
[![License](https://img.shields.io/github/license/yi-cheng-liu/basketball_registration)](./LICENSE.txt)
![Primary language](https://img.shields.io/github/languages/top/yi-cheng-liu/basketball_registration)


The title of our project is **Player Tracking in Bird-Eye Diagram Through Basketball Video**

In the sport of basketball, analyzing gameplay and opposing team strategies can be challenging due to tilted camera angles in footage. Instant playbacks offer coaches a better perspective, enabling them to assess match dynamics, make real-time adaptations, and analyze games more effectively. This enhances decision-making, team success, and fan engagement. **YOLOv5**, a popular multi-object detection model, along with robust court mapping with **ResNet18**, plays a crucial role in accurately identifying players and positioning them on the detected court.

Original           |  Board
:-----------------:|:---------------------:
![Original](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/demo_original.gif)  | ![Board](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/demo_board.gif)

**Related Paper:**

+ Maglo, Adrien, Astrid Orcesi, and Quoc-Cuong Pham. "KaliCalib: A Framework for Basketball Court Registration." Proceedings of the 5th International ACM Workshop on Multimedia Content Analysis in Sports. 2022.[**[PDF]**](https://arxiv.org/abs/2209.07795)

---

## 📚 1. Installation

Create a virtual environment:
```
virtualenv venv
source venv/bin/activate
```

Install the dependancies:
```
pip install -r requirements.txt
```

## ⚙️ 2. Run the program

Default run
```
python bird_eye_video.py
```

#### Add arguments
Change model
```
python bird_eye_video.py --modelPath models/model_challenge.pth
```
Change input video
```
python bird_eye_video.py --inputPath input/demo.mp4
```
Change weight
```
python bird_eye_video.py --weightPath yolov5/runs/train/exp2/weights/best.pt
```
Change floortexture
```
python bird_eye_video.py --floorTexturePath input/floor_texture/concrete.jpg
```




## 🛠️ 3. Dataset

The dataset used in this study includes bounding box locations of players and referees, as well as court vertex locations. Due to challenges in collecting data from actual NBA games, data was gathered from NBA 2k19, ensuring controlled scenes with visible sidelines. This dataset improves court detection, player tracking, and model generalization.

The self-annotated dataset is opensourced in [Roboflow](https://universe.roboflow.com/nba2kplayer/nba2k-player-model)




## 🌟 4. Results

### NBA 2K19 data

Original           |  Homography           |  Board
:-----------------:|:---------------------:|:---------------------:
![Original](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/demo_original.gif)  |  ![Homography](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/demo_homography.gif)  |  ![Board](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/demo_board.gif)

### Real Game data


Original           |  Homography           |  Board
:-----------------:|:---------------------:|:---------------------:
![Original](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/real_demo_original.gif)  |  ![Homography](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/real_demo_homography.gif)  |  ![Board](https://github.com/yi-cheng-liu/basketball_registration/blob/main/.assets/real_demo_board.gif)

## 🎬 5. Documentation

+ Project Document: [**[PDF]**](https://drive.google.com/file/d/1GAQ3sh8x2o-xqoOj1Emo5974rCg2dLBc/view?usp=share_link)



## 🏅 6. Reference

[1] P. Kaarthick, “An automated player detection and tracking in basketball game,” Computers, Materials Continua, vol. 58, pp. 625–639, 01 2019.

[2] D. Farin, S. Krabbe, P. With, and W. Effelsberg, “Robust camera calibration for sport videos using court models,” vol. 5307, 01 2004, pp. 80–91.

[3] W.-L. Lu, “Learning to track and identify players from broadcast sports videos,” Ph.D. dissertation, University of British Columbia, 2011. [**Link**](https://open.library.ubc.ca/collections/ubctheses/24/items/1.0052129)

[4] G. Jocher, “YOLOv5 by Ultralytics,” May 2020. [**Link**](https://github.com/ultralytics/yolov5)

[5] A. Maglo, A. Orcesi, and Q.-C. Pham, “Kalicalib: A framework for basketball court registration,” in Proceedings of the 5th International ACM Workshop on Multimedia Content Analysis in Sports, 2022, pp. 111–116. 

[6] M. Martinez, C. Sitawarin, K. Finch, L. Meincke, A. Yablonski, and A. Kornhauser, “Beyond grand theft auto v for training, testing and enhancing deep learning in self driving cars,” arXiv preprint arXiv:1712.01397, 2017.

[7] P. Dwivedi. (2019) March madness — analyze video to detect players, teams, and who attempted the basket. [**Link**](https://towardsdatascience.com/march-madness-analyze-video-to-detect-players-teams-and-who-attempted-the-basket-8cad67745b88)

[8] Stephan. (2019) Open source sports video analysis using machine learning. [**Link**](https://dev.to/stephan007/open-source-sports-video-analysis-using-maching-learning-2ag4)





## 📫 7. Contact

+ Tien-Li Lin, Email: tienli@umich.edu
+ Yi-Cheng Liu, Email: liuyiche@umich.edu
+ Wei-Cheng Chiang, Email: imarthur@umich.edu



**Please cite the author's paper if you use the code in your work.**

```
@inproceedings{maglo2022kalicalib,
  title={KaliCalib: A Framework for Basketball Court Registration},
  author={Maglo, Adrien and Orcesi, Astrid and Pham, Quoc-Cuong},
  booktitle={Proceedings of the 5th International ACM Workshop on Multimedia Content Analysis in Sports},
  pages={111--116},
  year={2022}
}
```

