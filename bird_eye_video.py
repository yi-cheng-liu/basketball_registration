#!/usr/bin/python3

from yolov5.detect import run
from modeling.example_camera_model import compute_camera_model, MEAN_H
from data.datasets.viewds import getFieldPoints
from model_resnet import makeModel
import sys
import os
import argparse
import random
import time
from tqdm import tqdm


import cv2
from cv2 import imshow
import torch
import numpy as np
import torchvision.transforms as transforms
from deepsport_utilities.calib import Calib
from calib3d.points import Point2D, Point3D

sys.path.append(".")

###############################################################################
#   Modified Parameters                                                       #
###############################################################################
colormap = {
    0: (0, 0, 255),      # home
    1: (0, 122, 83),      # away
    2: (255, 0, 0)       # referee
}

RESULT_DIR = 'results'
BOARD_DIR = 'boards'
HOMOGRAPHY_DIR = 'homographies'
ORIGINAL_DIR = 'original'

FRAME_OF_MOTION = 15

###############################################################################
#   End of Code                                                               #
###############################################################################

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
    points = np.expand_dims(getFieldPoints()[:, 0:2], axis=0)
    return points


def drawTemplateFigure(fieldPoints):
    s = 1/4  # scale
    m = 20  # margin

    H = np.float32([
        [s, 0, m],
        [0, s, m],
        [0, 0, 1],
    ])

    # Black background
    # floor_texture = np.ones((int(FIELD_WIDTH * s + m * 2),
    #               int(FIELD_LENGTH * s + m * 2), 3), dtype=np.uint8)

    floor_texture = cv2.resize(cv2.imread(
        args.floorTexturePath, cv2.IMREAD_COLOR), (740, 415))
    drawField(floor_texture, H, (255, 255, 255), 4)

    # pointsToDraw = cv2.perspectiveTransform(fieldPoints, H)[0].astype(int)[:-2]
    # for p in pointsToDraw:
    #     cv2.circle(img, (p[0], p[1]), 7, (0, 0, 255), -1)

    return floor_texture


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
    cv2.line(img, tuple(pointsToDraw[0]), tuple(
        pointsToDraw[1]), color, thickness)

    # sideline
    cv2.line(img, tuple(pointsToDraw[2]), tuple(
        pointsToDraw[0]), color, thickness)

    # middle box
    cv2.line(img, tuple(pointsToDraw[3]), tuple(
        pointsToDraw[4]), color, thickness)  # top and bottom line
    cv2.line(img, tuple(pointsToDraw[4]), tuple(
        pointsToDraw[5]), color, thickness)  # freethrow line

    # three point line, side straight
    cv2.line(img, tuple(pointsToDraw[6]), tuple(
        pointsToDraw[7]), color, thickness)
    cv2.line(img, tuple(pointsToDraw[1]), tuple(
        pointsToDraw[8]), color, thickness)

    # three point and free throw circle
    drawFieldCircle(img, H, mulCoef, translateCoef, color, thickness,
                    np.array([580, FIELD_WIDTH / 2]), 180, 0, np.pi / 2)
    drawFieldCircle(img, H, mulCoef, translateCoef, color, thickness, np.array(
        [157.5, FIELD_WIDTH / 2]), 675, 0, np.pi / 2 - 0.211)
    drawFieldCircle(img, H,  mulCoef, translateCoef, color, thickness, np.array(
        [FIELD_LENGTH / 2, FIELD_WIDTH / 2]), 180, np.pi / 2, np.pi)


def drawFieldCircle(img, H,  mulCoef, translateCoef,
                    color, thickness, center, radius, startAngle, stopAngle):
    # Draw small cicle
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

###############################################################################
#   Modified Code                                                             #
###############################################################################


def make_synthetic_view(img, corners, size):
    h = size[0]
    w = size[1]

    input_pts = np.float32(corners)
    output_pts = np.float32([[0, 0],
                            [w - 1, 0],
                            [w - 1, h - 1],
                            [0, h - 1]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    out = cv2.warpPerspective(src=img, M=M, dsize=(int(w), int(h)))
    return out, M


def do_homography(img, H, positions, labels, is_leftfield):

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
        corners = np.array([topPoints[0], topPoints[1],
                           botPoints[1], botPoints[0]])
    else:
        corners = np.array([topPoints[1], topPoints[0],
                           botPoints[0], botPoints[1]])

    size = (FIELD_WIDTH/5, FIELD_LENGTH/5)

    result, M = make_synthetic_view(img, corners, size)

    for i, xy in enumerate(positions):

        # point transformation
        p = np.array([xy[0], xy[1], 1])

        p1 = M @ p
        p1 = p1/p1[2]
        xy = (int(p1[0]), int(p1[1]))
        cv2.circle(result, xy, 1, colormap[int(labels[i])], 10)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # cv2.imshow("board", result)
    # cv2.waitKey(1)

    return M, result


def drawPlayer(players_position_of_frames, players_position, labels, M, is_leftfield, templateCourtImg):
    s = 1/4  # scale
    m = 20  # margin

    if is_leftfield:
        rightfield_offset = 0
    else:
        rightfield_offset = int(FIELD_LENGTH * s/2)

    y_ratio = (FIELD_WIDTH * s)/(FIELD_WIDTH / 5)
    x_ratio = (FIELD_LENGTH * s/2)/(FIELD_LENGTH / 5)

    newest_player_position = []

    # Append the position after homography
    for i, xy in enumerate(players_position):
        pos = np.array([xy[0], xy[1], 1])
        pos_homography = M @ pos
        pos_homography = pos_homography/pos_homography[2]
        pos_homography = (int(m+pos_homography[0]*x_ratio+rightfield_offset),
                          int(m+pos_homography[1]*y_ratio))
        players_position_of_frames.append(pos_homography)
        newest_player_position.append(pos_homography)

    # draw history player trajectory
    for i, xy in enumerate(players_position_of_frames):
        cv2.circle(templateCourtImg, (xy[0], xy[1]),
                   4, colormap[int(labels[i])], -1)

    # draw the newest player position
    for i, xy in enumerate(newest_player_position):
        pos_len = len(newest_player_position)
        color = tuple(int(0.56 * c) for c in colormap[int(labels[-pos_len+i])])
        cv2.circle(templateCourtImg, (xy[0], xy[1]), 4, color, -1)

    # print(len(newest_player_position))
    newest_player_position.clear()

    templateCourtImg = cv2.cvtColor(templateCourtImg, cv2.COLOR_BGR2RGB)

    # Show the current frame
    # cv2.imshow("template", templateCourtImg)
    # cv2.waitKey(0)
    return templateCourtImg, players_position_of_frames


###############################################################################
#   End of Code                                                               #
###############################################################################


def getModel(modelPath):
    # device = torch.device("cuda")
    device = torch.device("cpu")
    if True:
        model = makeModel().to(device)
    else:
        model = convnext_tiny().to(device)

    # load the model
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    print("Model number of parameters:", sum(p.numel()
          for p in model.parameters()))

    return model, device


def estimateCalib(model, device, fieldPoints2d, oriImg, visualization, positions, labels, is_leftfield):
    oriHeight, oriWidth = oriImg.shape[0:2]
    npImg = cv2.resize(oriImg, (IMG_WIDTH, IMG_HEIGHT))

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.3235, 0.5064, 0.4040],
        #                     std=[0.1718, 0.1565, 0.1687])
    ])
    img = tf(npImg).to(device).unsqueeze(0)

    with torch.no_grad():
        heatmaps = model(img)

    return estimateCalibHM(heatmaps, npImg, fieldPoints2d, oriHeight, oriWidth, visualization, positions, labels, is_leftfield)


def estimateCalibHM(heatmaps, npImg, fieldPoints2d, oriHeight, oriWidth, visualization, positions, labels, is_leftfield):

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


###############################################################################
#   Modified Code                                                             #
###############################################################################
    max_heatmap_point = 0
###############################################################################
#   End of Code                                                               #
###############################################################################

    for i in range(nbPoints):

        if True:
            # maxVal = np.max(out[i])
            # print(i, maxVal)

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

        max_heatmap_point = i
        # cv2.circle(npImg, (pImg[1], pImg[0]), 3, (255, 0, 0), -1)
        # cv2.putText(npImg, str(i), (pImg[1], pImg[0] - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        srcPoints.append(fieldPoints2d[0][i])
        dstPoints.append(p[::-1])

###############################################################################
#   Modified Code                                                             #
###############################################################################

    if max_heatmap_point > 85:
        is_leftfield = False
    else:
        is_leftfield = True


###############################################################################
#   End of Code                                                               #
###############################################################################

    calib = Calib.from_P(np.array(MEAN_H), width=oriWidth, height=oriHeight)

    if len(srcPoints) >= 4:

        Hest, keptPoints = cv2.findHomography(
            np.array(srcPoints),
            np.array(dstPoints),
            cv2.RANSAC,
            # 0,
            ransacReprojThreshold=50,
            maxIters=2000
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

            newCalib = compute_camera_model(
                dstPoints2d, srcPoints3d, (oriHeight, oriWidth))
            if checkProjection(newCalib, oriWidth, oriHeight):
                calib = newCalib
    else:
        Hest = None

    if Hest is not None:

        pass
        drawField(npImg, Hest, (0, 0, 255), 6)

        M, result = do_homography(
            npImg, Hest, positions, labels, is_leftfield)

    else:
        print("Default calib")

    # drawCalibCourt(calib, npImg)

    if visualization:
        npImg = cv2.cvtColor(npImg, cv2.COLOR_BGR2RGB)
        # cv2.imshow("test", npImg)

        # cv2.waitKey(1)

    return calib, M, is_leftfield, result, npImg


def checkProjection(calib, imgWidth, imgHeight):
    points = np.float32([
        [imgWidth / 4, imgHeight / 2],
        [3 * imgWidth / 4, imgHeight / 2],
    ])

    points = Point2D(points.T)
    proj = calib.project_2D_to_3D(points, Z=0).T

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

    # h_pred
    heatmap = (heatmap*255).copy().astype(np.uint8)
    (cnts, _) = cv2.findContours(
        heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        M = cv2.moments(heatmap[target[1]: target[1] +
                        target[3], target[0]: target[0] + target[2]])
        # calculate x,y coordinate of center
        assert (M["m00"] > 0)
        cy, cx = M["m01"] / M["m00"] + \
            target[1], M["m10"] / M["m00"] + target[0]

    return cx, cy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate the homography of a sport field")
    parser.add_argument("--modelPath", default='models/model_challenge.pth',
                        help="Model path")
    parser.add_argument("--inputPath", default='input/video/demo.mp4',
                        help="Input video path")
    parser.add_argument("--weightPath", default='yolov5/runs/train/exp2/weights/best.pt',
                        help="Weight path")
    parser.add_argument("--floorTexturePath", default='input/floor_texture/wood.jpg',
                        help="'wood', 'concrete' and 'blue_paint are available")
    args = parser.parse_args()

    model, device = getModel(args.modelPath)

    # Time
    start_time = time.time()

    # Extract the file name from the file path, and set the result path
    input_video_path = args.inputPath
    file_name = os.path.basename(input_video_path)
    output_board_path = os.path.join(RESULT_DIR, BOARD_DIR,
                                     file_name[:-4] + '_board.mp4')
    output_homography_path = os.path.join(RESULT_DIR, HOMOGRAPHY_DIR,
                                          file_name[:-4] + '_homography.mp4')
    output_original_path = os.path.join(RESULT_DIR, ORIGINAL_DIR,
                                        file_name[:-4] + '_original.mp4')

    # Extract the directory path, and create it if necessary
    board_dir_path = os.path.dirname(output_board_path)
    homography_dir_path = os.path.dirname(output_homography_path)
    original_dir_path = os.path.dirname(output_original_path)
    if not os.path.exists(board_dir_path):
        os.makedirs(board_dir_path)
    if not os.path.exists(homography_dir_path):
        os.makedirs(homography_dir_path)
    if not os.path.exists(original_dir_path):
        os.makedirs(original_dir_path)

    # Information of the video
    vidcap = cv2.VideoCapture(input_video_path)
    board_out = cv2.VideoWriter(output_board_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                10, (740, 415))
    homo_out = cv2.VideoWriter(output_homography_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                               10, (560, 300))
    original_out = cv2.VideoWriter(output_original_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   10, (960, 540))
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, oriImg = vidcap.read()

    # Get the data of the field, and draw the template figure
    fieldPoints2d = getFieldPoints2d()
    fieldPoints3d = getFieldPoints()

    # Draw the parameters when don't want players'trajectory
    # templateCourtImg = drawTemplateFigure(fieldPoints2d)

    # parameters
    frame_idx = 0
    frame_count = 0
    frame_count_list = []
    players_position_of_frames = []
    labels_of_frames = []

    for frame_idx in tqdm(range(int(video_length/2))):
        success, oriImg = vidcap.read()
        if not success:
            break

        # run only accpet path of image, can be modified later
        cv2.imwrite('./tmp.jpg', oriImg)
        players_position, labels = run(weights=args.weightPath,
                                       exist_ok=True,
                                       name=file_name,
                                       source='./tmp.jpg')

        # Grayscale the image and reshape the image
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        x_gain, y_gain = IMG_WIDTH/oriImg.shape[1], IMG_HEIGHT/oriImg.shape[0]

        # store players position and change to the same scale as the field
        for j, player_xy in enumerate(players_position):
            players_position[j] = np.array([player_xy[0] * x_gain,
                                            player_xy[1] * y_gain])
        # draw points on the calibration map
        is_leftfield = True
        calib, M, is_leftfield, result, originalImg = estimateCalib(model, device, fieldPoints2d, oriImg,
                                                                    True, players_position, labels,
                                                                    is_leftfield)

        # Store n frames of players data, and thus will have a motion effect
        labels_of_frames += labels
        if frame_idx >= FRAME_OF_MOTION and frame_idx != 0:
            frame_count = frame_count_list[0]
            frame_count_list.append(len(players_position))
            frame_count_list.pop(0)
            while (frame_count > 0):
                players_position_of_frames.pop(0)
                labels_of_frames.pop(0)
                frame_count -= 1
        elif frame_idx < FRAME_OF_MOTION:
            frame_count_list.append(len(players_position))

        # Draw on the board image
        templateCourtImg = drawTemplateFigure(fieldPoints2d)  # no player court
        templateCourtImg = cv2.cvtColor(templateCourtImg, cv2.COLOR_BGR2RGB)

        BirdEyeCourtImg, players_position_of_frames = drawPlayer(players_position_of_frames,
                                                                 players_position,
                                                                 labels_of_frames, M,
                                                                 is_leftfield, templateCourtImg)
        # Output and save the result
        board_out.write(BirdEyeCourtImg)
        homo_out.write(result)
        original_out.write(originalImg)

    board_out.release()
    homo_out.release()
    original_out.release()

    elapsed_time = time.time() - start_time
    print("Result: ")
    print("   board saved in:      ", output_board_path)
    print("   homography saved in: ", output_homography_path)
    print("   original saved in:   ", output_original_path)
    print(f"Run time: {elapsed_time} seconds")
