import cv2
import numpy as np
import time

RESCALE_FACTOR = 1
HEIGHT = 360
WIDTH = 640

lower_red = np.array([0,  120, 120])
upper_red = np.array([20, 255, 255])
lower_table = np.array([0,  0, 128])
upper_table = np.array([180, 55, 255])

def main():
	cap = cv2.VideoCapture('output.avi')
	cap.set(3, int(WIDTH*RESCALE_FACTOR))
	cap.set(4, int(HEIGHT*RESCALE_FACTOR))	

	while 1:
		#---------SETUP--------
		render_time = time.time()

		ret, orig = cap.read()
		if not ret:
			#print("Empty frame, resetting")
			cap.set(1, 0)
			continue

		#---------DETECTING TABLE--------
		hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
		mask_table = cv2.inRange(hsv,lower_table, upper_table)
		_, contours, _ = cv2.findContours(mask_table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
		_, cnt, _ = getBiggestContour(contours)
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		intbox = np.int0(box)

		#--------CROPPING IMAGE--------
		w,h = int(rect[1][0]), int(rect[1][1])
		pts2 = np.float32([[0,h], [0,0], [w, 0], [w, h]])
		M = cv2.getPerspectiveTransform(box, pts2)
		cropped = cv2.warpPerspective(orig, M, (w, h))

		#-------DETECTING PUCK--------

		#-------CALCULATING PUCK TRAJECTORY------

		#-------SHOWING DEBUG INFO------
		frame = orig.copy()
		cv2.drawContours(frame, [intbox], 0, (255,255,0), 1)
		frame_info_text = "{}/{}".format(int(cap.get(1)),int(cap.get(7)))
		render_time = time.time() - render_time
		cv2.putText(frame, frame_info_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 1, cv2.LINE_AA)
		cv2.putText(frame, "RTime: {}ms".format(int(render_time*1000)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 1, cv2.LINE_AA)
		cv2.putText(frame, "-> FPS: {}".format(int(1/render_time)), (5, 75), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 1, cv2.LINE_AA)

		cv2.imshow("Original", frame)
		cv2.imshow("Croped", cropped)

		if cv2.waitKey(1) & 0xFF is ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def getBiggestContour(cnts):
	lenCnts = len(cnts)
	if lenCnts == 0:
		return False, False, False
	elif lenCnts == 1:
		return True, cnts[0], cv2.contourArea(cnts[0])
	else:
		biggestCnt = None
		biggestArea = -1

		for cnt in cnts:
			area = cv2.contourArea(cnt)
			if biggestArea == -1 or area > biggestArea:
				biggestArea = area
				biggestCnt = cnt
		return True, biggestCnt, biggestArea

if __name__ == "__main__":
	main()