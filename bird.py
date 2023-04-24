#!/usr/bin/python3

import sys
import os
import argparse
import glob
import subprocess
import random

import cv2
from cv2 import imshow
import torch
import numpy as np
import torchvision.transforms as transforms
from deepsport_utilities.calib import Calib
from calib3d.points import Point2D, Point3D
from scipy.spatial.distance import cdist

sys.path.append(".")
from model_resnet import makeModel
from data.datasets.viewds import getFieldPoints
from modeling.example_camera_model import compute_camera_model, MEAN_H
#################################################
from yolov5.detect import run
colormap = {
        0:(255,255,255),
        1:(0,0,255),
        2:(255,0,0) 
    }
RESULT_DIR='results'	
BOARD_DIR='boards'	
HOMO_DIR='homographies'	
#################################################
seed = 4212

IMG_WIDTH = 960
IMG_HEIGHT = 540

FIELD_LENGTH = 2800
FIELD_WIDTH = 1500

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def getFieldPoints2d():
    points = np.expand_dims(getFieldPoints()[:,0:2], axis=0)
    return points


def drawField(img, H, color, thickness):
    mulCoefs = [[1, 1],
                [1, -1],
                [-1, -1],
                [-1, 1]]
    translateCoefs = [[0, 0],
                      [0, FIELD_WIDTH],
                      [FIELD_LENGTH, FIELD_WIDTH],
                      [FIELD_LENGTH, 0]]
    
    for mulCoef, translateCoef in zip(mulCoefs, translateCoefs):
        drawQuarterField(img, H, color, thickness, mulCoef, translateCoef)
        
    return


def drawQuarterField(img, H, color, thickness, mulCoef, translateCoef):

    fieldPoints = np.float32([[
        [0, 0],
        [FIELD_LENGTH / 2, 0],
        [0, FIELD_WIDTH / 2],

        [0, 505],
        [580, 505],
        [580, 750],

        [0, 90],
        [299, 90],

        [FIELD_LENGTH / 2, FIELD_WIDTH / 2],
    ]])
    

    fieldPoints *= mulCoef
    fieldPoints += translateCoef

    pointsToDraw = cv2.perspectiveTransform(fieldPoints, H)[0].astype(int)
    
    # top and bottom line
    cv2.line(img, tuple(pointsToDraw[0]), tuple(pointsToDraw[1]), color, thickness)
    #cv2.line(img, tuple(pointsToDraw[2]), tuple(pointsToDraw[1]), color, thickness)
    # sideline 
    
    cv2.line(img, tuple(pointsToDraw[2]), tuple(pointsToDraw[0]), color, thickness)
    
    # middle box
    cv2.line(img, tuple(pointsToDraw[3]), tuple(pointsToDraw[4]), color, thickness) # top and bottom line
    cv2.line(img, tuple(pointsToDraw[4]), tuple(pointsToDraw[5]), color, thickness) # freethrow line
    
    # three point line, side straight
    cv2.line(img, tuple(pointsToDraw[6]), tuple(pointsToDraw[7]), color, thickness)
    cv2.line(img, tuple(pointsToDraw[1]), tuple(pointsToDraw[8]), color, thickness)
    
    # three point and free throw circle 
    drawFieldCircle(img, H, mulCoef, translateCoef, color, thickness, np.array([580, FIELD_WIDTH / 2]), 180, 0, np.pi / 2)
    drawFieldCircle(img, H, mulCoef, translateCoef, color, thickness, np.array([157.5, FIELD_WIDTH / 2]), 675, 0, np.pi / 2 - 0.211)
    drawFieldCircle(img, H,  mulCoef, translateCoef, color, thickness, np.array([FIELD_LENGTH / 2, FIELD_WIDTH / 2]), 180, np.pi / 2, np.pi)

#=======================================================
def make_synthetic_view(img, corners, size):
    h=size[0]
    w=size[1]
    
    input_pts = np.float32(corners)
    output_pts = np.float32([[0, 0],
                            [w - 1,0],
                            [w  - 1, h  - 1],
                            [0, h  - 1]])
    
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(src=img,M=M,dsize=(int(w),int(h)))
    return out, M

def do_homography(img, H, positions, labels,is_leftfield,clip_path,frame_name ):	
    # Identify the field and change the view
    if is_leftfield:	
        mulCoefs = [[1, 1],	
                    [1, -1]]	
        translateCoefs = [[0, 0],	
                          [0, FIELD_WIDTH]]	
    	
    else:	
        mulCoefs = [[-1, 1],	
                    [-1, -1]]	
        translateCoefs = [[FIELD_LENGTH, 0],	
                          [FIELD_LENGTH, FIELD_WIDTH]]	
        	
    fieldPoints = np.float32([[	
        [0, 0],	
        [FIELD_LENGTH / 2, 0],	
        [0, FIELD_WIDTH / 2],

        [0, 505],	
        [580, 505],	
        [580, 750],	

        [0, 90],	
        [299, 90],	
        [FIELD_LENGTH / 2, FIELD_WIDTH / 2],	
    ]])	
    	
     	
    topPoints = fieldPoints*mulCoefs[0]	
    topPoints += translateCoefs[0]	
    topPoints = cv2.perspectiveTransform(topPoints, H)[0].astype(int)	
        	
    	
    botPoints = fieldPoints*mulCoefs[1]	
    botPoints += translateCoefs[1]	
    botPoints = cv2.perspectiveTransform(botPoints, H)[0].astype(int)	
    	
    if is_leftfield:	
        corners=np.array([topPoints[0],topPoints[1],botPoints[1],botPoints[0]])	
    else:	
        corners=np.array([topPoints[1],topPoints[0],botPoints[0],botPoints[1]])	
        	
    size=(FIELD_WIDTH/5,FIELD_LENGTH/5)	
    result, M = make_synthetic_view(img, corners, size)	
    	
    	
    	
    for i,xy in enumerate(positions):	
        	
        #point transformation	
        p = np.array([xy[0],xy[1],1])	
        p1 = M @ p	
        p1 = p1 / p1[2]	
        xy = (int(p1[0]),int(p1[1]))	
        cv2.circle(result, xy, 1, colormap[int(labels[i])], 10)	
    	
    	
    	
    result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)	
    cv2.imshow("board",result)	
    	
    write_path=os.path.join(RESULT_DIR, HOMO_DIR, frame_name)	
        	
    cv2.waitKey(1)	
        	
    return M,result
    
#=======================================================     

def drawFieldCircle(img, H,  mulCoef, translateCoef,
        color, thickness, center, radius, startAngle, stopAngle):
    #Draw small cicle
    fieldPoints = []
    for teta in np.arange(startAngle, stopAngle, 0.2 * np.pi / 180):
        p = center + np.array([radius * np.cos(teta), radius * np.sin(teta)])
        fieldPoints.append(p)

    fieldPoints = np.array([fieldPoints])
    fieldPoints *= mulCoef
    fieldPoints += translateCoef

    pointsToDraw = cv2.perspectiveTransform(fieldPoints, H)[0].astype(int)

    lastPoint = None
    for p in pointsToDraw:
        if lastPoint is not None:
            cv2.line(img, tuple(lastPoint), tuple(p), color, thickness)
        lastPoint = p


def drawTemplateFigure(fieldPoints, player_position_3D, labels, M, is_leftfield, clip_path, frame_name):
    s = 1/4 #scale
    m = 20 #margin

    H = np.float32([
        [s, 0, m],
        [0, s, m],
        [0, 0, 1],
        ])
    
    
    img = np.ones((int(FIELD_WIDTH * s + m * 2), int(FIELD_LENGTH * s + m * 2), 3), dtype = np.uint8) * 255
    drawField(img, H, (255, 0, 0), 5, positions, labels)
    for i, player in enumerate(player_position_3D):
        cv2.circle(img, (int(player[1]), int(player[0])), 5, (0, 0, 255), -1)


    # Heat 
    pointsToDraw = cv2.perspectiveTransform(fieldPoints, H)[0].astype(int)[:-2]
    # for p in pointsToDraw:
    #     cv2.circle(img, (p[0], p[1]), 7, (0, 0, 255), -1)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imshow("template", img)
    cv2.waitKey(1)


def drawCalibCourt(calib, img):
    points = np.float32([
        [0, 0, 0],
        [FIELD_LENGTH, 0, 0],
        [FIELD_LENGTH, FIELD_WIDTH, 0],
        [0, FIELD_WIDTH, 0],
    ])

    points = Point3D(points.T)
    proj = calib.project_3D_to_2D(points).T.astype(int)

    color = (255, 255, 0)
    thickness = 2

    cv2.line(img, proj[0], proj[1], color, thickness)
    cv2.line(img, proj[1], proj[2], color, thickness)

    cv2.line(img, proj[2], proj[3], color, thickness)
    cv2.line(img, proj[3], proj[0], color, thickness)

    return True


def getModel(modelPath):
    # device = torch.device("cuda")
    device = torch.device("cpu")
    if True:
        model = makeModel().to(device)
    else:
        model= convnext_tiny().to(device)
        
    # load the model
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    print("Model number of parameters:", sum(p.numel() for p in model.parameters()))

    return model, device


def estimateCalib(model, device, fieldPoints2d, fieldPoints3d, oriImg, visualization, player_position_3D, labels ):
    oriHeight, oriWidth = oriImg.shape[0:2]
    npImg = cv2.resize(oriImg, (IMG_WIDTH, IMG_HEIGHT))

    tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            #transforms.Normalize(mean=[0.3235, 0.5064, 0.4040],
            #                     std=[0.1718, 0.1565, 0.1687])
        ])
    img = tf(npImg).to(device).unsqueeze(0)

    with torch.no_grad():
        heatmaps = model(img)

    return estimateCalibHM(heatmaps, npImg, fieldPoints2d, fieldPoints3d, oriHeight, oriWidth, visualization, player_position_3D, labels)


def estimateCalibHM(heatmaps, npImg, fieldPoints2d, fieldPoints3d, oriHeight, oriWidth, visualization, player_position_3D, labels ):

    out = heatmaps[0].cpu().numpy()
    kpImg = out[-1].copy()
    kpImg /= np.max(kpImg)

    # if visualization:
        # cv2.imshow("kpImg", kpImg)

    srcPoints = []
    dstPoints = []
    nbPoints = out.shape[0] - 1 - 2

    pixelScores = np.swapaxes(out, 0, 2)
    pixelMaxScores = np.max(pixelScores, axis=2, keepdims=2)

    pixelMax = (pixelScores == pixelMaxScores)
    pixelMap = np.swapaxes(pixelMax, 0, 2).astype(np.uint8)
    pixelMap = (out > 0.82) * pixelMap

    # pixelMap = (out > 0) * pixelMap
    # pixelMap = (out > 0.7) * out
    # pixelMap = (out > 0.95) * pixelMap
    
    sideline_left = sideline_right = sideline_top = sideline_bottom = np.empty(shape=[0, 3])

    for i in range(nbPoints):
        if True:
            #maxVal = np.max(out[i])
            #print(i, maxVal)

            M = cv2.moments(pixelMap[i])
            # calculate x,y coordinate of center
            if M["m00"] == 0:
                continue
            p = (M["m01"] / M["m00"], M["m10"] / M["m00"])
        else:
            p = find_coord(pixelMap[i])
            if p is None:
                continue
            p = p[1], p[0]

        p *= np.array([4, 4])
        pImg = [round(p[0]), round(p[1])]
        
        #cv2.circle(npImg, (pImg[1], pImg[0]), 3, (255, 0, 0), -1)
        #cv2.putText(npImg, str(i), (pImg[1], pImg[0] - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        
        # print out the image with the points
        # print(i, ": ", pImg)
        
###############################################################################
#   My Code                                                                   #
###############################################################################
        
        # low sideline 
        if (0 <= i <= 12):
            sideline_bottom = sideline_points(i, pImg, sideline_bottom)
            
        # top sideline
        if (78 <= i <= 90):
            sideline_top = sideline_points(i, pImg, sideline_top)
        
        # Left sideline
        if ((i % 13) == 0):
            sideline_left = sideline_points(i, pImg, sideline_left)
        
        # Right sideline 
        if (((i+1) % 13) == 0):
            sideline_right = sideline_points(i, pImg, sideline_right)
        
###############################################################################
#   My Code                                                                   #
###############################################################################

        srcPoints.append(fieldPoints2d[0][i])
        dstPoints.append(p[::-1])
    
    sideline = {'sideline_top':sideline_top, 
                'sideline_bottom': sideline_bottom, 
                'sideline_left': sideline_left, 
                'sideline_right' : sideline_right}
    
    calib = Calib.from_P(np.array(MEAN_H), width=oriWidth, height=oriHeight)
    if len(srcPoints) >= 4:
        Hest, keptPoints = cv2.findHomography(
            np.array(srcPoints),
            np.array(dstPoints),
            cv2.RANSAC,
            #0,
            ransacReprojThreshold = 50,
            maxIters = 2000
        )

        if Hest is not None:
            srcPoints3d = []
            dstPoints2d = []
            for i, kept in enumerate(keptPoints):
                if kept:
                    srcPoints3d.append(np.concatenate((srcPoints[i], [0])))
                    dstPoints2d.append(dstPoints[i])
                    p = dstPoints[i][::-1]
                    # cv2.circle(npImg, (round(p[1]), round(p[0])), 5, (0, 0, 255), -1)

            newCalib = compute_camera_model(dstPoints2d, srcPoints3d, (oriHeight, oriWidth))
            if checkProjection(newCalib, oriWidth, oriHeight):
                calib = newCalib
    else:
        Hest = None

    if Hest is not None:
        pass
        drawField(npImg, Hest, (0, 0 , 255), 6 )
        
        do_homography(npImg, Hest, player_position_3D, labels )


    else:
        print("Default calib")
        
###############################################################################
#   My Code                                                               #
###############################################################################
    
    #draw_courtline(npImg, sideline['sideline_top'])
    # draw_courtline(npImg, sideline['sideline_bottom'])
    # draw_courtline(npImg, sideline['sideline_left'])
    # draw_courtline(npImg, sideline['sideline_right'])
    
###############################################################################
#   My Code                                                                   #
###############################################################################

    # drawCalibCourt(calib, npImg)

    if visualization:
        npImg=cv2.cvtColor(npImg, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", npImg)
        cv2.waitKey(1)

    return calib, sideline


def calcAngle(v1, v2):

    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_vector_1, unit_vector_2)

    return np.arccos(dot_product) * 360 / np.pi


def checkProjection(calib, imgWidth, imgHeight):
    points = np.float32([
        [imgWidth / 4, imgHeight / 2],
        [3 * imgWidth / 4, imgHeight / 2],
    ])

    points = Point2D(points.T)
    proj = calib.project_2D_to_3D(points, Z = 0).T

    if np.linalg.norm(proj[1] - proj[0]) > 1800:
        print("wrong 1")
        return False

    if np.linalg.norm(proj[1] - proj[0]) < 100:
        print("wrong 2")
        return False

    return True


def find_coord(heatmap):
    if np.max(heatmap) <= 0:
        return None

    #h_pred
    heatmap = (heatmap*255).copy().astype(np.uint8)
    (cnts, _) = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
    for i in range(len(rects)):
        area = rects[i][2] * rects[i][3]
        if area > max_area:
            max_area_idx = i
            max_area = area
    target = rects[max_area_idx]
    if False:
        cx = target[0] + target[2] / 2
        cy = target[1] + target[3] / 2

    else:
        M = cv2.moments(heatmap[target[1] : target[1] + target[3], target[0] : target[0] + target[2]])
        # calculate x,y coordinate of center
        assert(M["m00"] > 0)
        cy, cx = M["m01"] / M["m00"] + target[1], M["m10"] / M["m00"] + target[0]


    return cx, cy



###############################################################################
#   My Function                                                               #
###############################################################################

def sideline_points(i, pImg, sideline):
    pImg_point = np.array(pImg).reshape(1, 2)
    pImg_point = np.hstack((np.array(i).reshape(1,1), pImg_point))
    sideline = np.vstack((sideline, pImg_point))
    
    return sideline

def draw_courtline(img, points):
    print('sideline_top:')
    print(points)
    
    # Define the x and y variables
    x_points = points[:, 1]
    y_points = points[:, 2]

    # Find the coefficients of the least squares line
    A = np.vstack([x_points, np.ones(len(x_points))]).T
    coefficients = np.linalg.lstsq(A, y_points, rcond=None)[0]

    # Create an array of the predicted y-values
    predicted_y = coefficients[0] * x_points + coefficients[1]

    # Draw the least squares line on the image
    start_point = (int(coefficients[0] * x_points.min() + coefficients[1]), int(x_points.min()))
    end_point = (int(coefficients[0] * x_points.max() + coefficients[1]), int(x_points.max()))
    print(start_point, end_point)
    color = (0, 0, 255) # BGR format
    thickness = 2
    cv2.line(img, start_point, end_point, color, thickness)

###############################################################################
#   End of My Function                                                               #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the homography of a sport field")
    parser.add_argument("modelPath", help="Model path")
    parser.add_argument("inputPath", help="Input image path or directory")

    args = parser.parse_args()

    inputPath = args.inputPath
    print(inputPath)
    # model, device = getModel(args.modelPath)
    
    tmp = inputPath
    i = 1
    for i in range(180):
        inputPath = tmp
        # inputPath += "img%d.png" % i
        inputPath += "ezgif-frame-%03d.jpg" % i
        # inputPath += "img1.jpg"
        print(inputPath)
        
        if os.path.isfile(inputPath):
            
            
            model, device = getModel(args.modelPath)
            fieldPoints2d = getFieldPoints2d()
            fieldPoints3d = getFieldPoints()
            oriImg = cv2.imread(inputPath)
            oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
            
            
            player_position_3D, labels=run(weights='yolov5/runs/train/exp/weights/best.pt',source=inputPath)
            
            print('oriImg',oriImg.shape)
            # Draw on the original image
            '''
            for i,xy in enumerate(player_position_3D):
                
                xy=(int(xy[0]),int(xy[1]))
                print("x: ", xy[0])
                print("y: ", xy[1])
                cv2.circle(oriImg, xy, 1, colormap[int(labels[i])], 50)
            '''    
            
            # find the gain of the x and y
            x_gain, y_gain = IMG_WIDTH/oriImg.shape[1], IMG_HEIGHT/oriImg.shape[0]
            
            for i,xy in enumerate(player_position_3D):
                player_position_3D[i] = np.array([xy[0]*x_gain,xy[1]*y_gain])
                print("After resize: ", (player_position_3D[i].astype(int)))

            
              
            calib, sideline = estimateCalib(model, device, fieldPoints2d, fieldPoints3d, oriImg, True, player_position_3D, labels )            
            
            ###################################################################
            # My Code                                                         #
            ###################################################################
