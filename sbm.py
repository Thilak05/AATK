import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time
import openpyxl
from PIL import Image

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Heart Rate Data"
ws.append(["Timestamp", "Heart Rate (BPM)", "Image Path"])

realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

webcam = cv2.VideoCapture(0)
detector = FaceDetector()

webcam.set(3, realWidth)
webcam.set(4, realHeight)

levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

plotY = LivePlot(realWidth, realHeight, [60, 120], invert=True)

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth // 2, 40)
fpsTextLocation = (500, 600)

fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

bpmCalculationFrequency = 10  # 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

save_interval = 10 
last_saved_time = time.time()

exit_time = 70
start_time = time.time()

i = 0
ptime = 0
ftime = 0
bpm_values = []

while (True):
    ret, frame = webcam.read()
    if ret == False:
        break

    frame, bboxs = detector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        fourierTransform[mask == False] = 0

        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
        frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()
        bpm_values.append(bpm_value)
        imgPlot = plotY.update(float(bpm_value))

        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value}', bpmTextLocation, scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

        current_time = time.time()
        if (current_time - last_saved_time) >= save_interval:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = f"datas/heart_rate_image_{timestamp}.jpg"
            cv2.imwrite(image_path, frameDraw)
            last_saved_time = current_time

            ws.append([timestamp, bpm_value, image_path])
            wb.save("datas/heart_rate_data.xlsx")

    if (time.time() - start_time) >= exit_time:
        break

average_bpm = np.mean(bpm_values)
timestamp = time.strftime("%Y%m%d_%H%M%S")

ws.append([timestamp, average_bpm, "Average BPM"])
wb.save("datas/heart_rate_data.xlsx")

webcam.release()
