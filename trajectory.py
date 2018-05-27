import cv2
import numpy as np
import time
import math

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

	last_valid_puck_location = False
	last_render_time = 1

	while 1:
		#---------SETUP--------
		render_time = time.time()

		ret, orig = cap.read()
		if not ret:
			#print("Empty frame, resetting")
			cap.set(1, 0)
			continue

		#---------DETECTING TABLE--------
		hsv_table = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
		mask_table = cv2.inRange(hsv_table,lower_table, upper_table)
		_, contours, _ = cv2.findContours(mask_table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
		_, cnt, _ = getBiggestContour(contours)
		rect_table = cv2.minAreaRect(cnt)
		box_table = cv2.boxPoints(rect_table)

		#--------CROPPING IMAGE--------
		w,h = int(rect_table[1][0]), int(rect_table[1][1])
		pts2 = np.float32([[0,h], [0,0], [w, 0], [w, h]])
		M = cv2.getPerspectiveTransform(box_table, pts2)
		cropped = cv2.warpPerspective(orig, M, (w, h))

		#-------DETECTING PUCK-------- (currently by color detecting, maybe again using contours?)
		hsv_puck = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
		mask_puck = cv2.inRange(hsv_puck, lower_red, upper_red)
		res_puck = cv2.bitwise_and(cropped, cropped, mask=mask_puck)
		blur_puck = cv2.GaussianBlur(res_puck, (15,15), 0)
		_, puck = cv2.threshold(blur_puck, 70, 255, cv2.THRESH_BINARY)

		puck_xs = np.where(puck != 0)[1]
		puck_ys = np.where(puck != 0)[0]
		puck_xstd = np.std(puck_xs)
		puck_ystd = np.std(puck_xs)
		puck_x = np.mean(puck_xs)
		puck_y = np.mean(puck_xs) # possibly if std is too high, we can consider that we have a problem?
		puck_xs = [x for x in puck_xs if x <= puck_x+puck_xstd or x >= puck_x-puck_xstd]
		puck_ys = [y for y in puck_ys if y <= puck_y+puck_ystd or y >= puck_y-puck_ystd]
		puck_x = np.mean(puck_xs)
		puck_y = np.mean(puck_ys)

		#-------CALCULATING PUCK TRAJECTORY------
		speed = -1
		if last_valid_puck_location is False:
			if not (math.isnan(puck_x) or math.isnan(puck_y)):
				last_valid_puck_location = (puck_x, puck_y)
		elif not (math.isnan(puck_x) or math.isnan(puck_y)):
			#....
			change = math.sqrt((puck_x-last_valid_puck_location[0])**2 + (puck_y-last_valid_puck_location[1])**2)
			speed = change / last_render_time
			last_valid_puck_location = (puck_x, puck_y)

		#-------SHOWING DEBUG INFO------
		frame = cropped.copy()

		if not (math.isnan(puck_x) or math.isnan(puck_y)):
			cv2.line(frame, (int(puck_x), int(puck_y)),(int(puck_x), int(puck_y)), (255,0,0),10) 

		frame_info_text = "{}/{}".format(int(cap.get(1)),int(cap.get(7)))
		render_time = time.time() - render_time
		last_render_time = render_time
		cv2.putText(frame, frame_info_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1, cv2.LINE_AA)
		cv2.putText(frame, "RTime: {}ms".format(int(render_time*1000)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1, cv2.LINE_AA)
		cv2.putText(frame, "-> FPS: {}".format(int(1/render_time)), (5, 75), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1, cv2.LINE_AA)
		cv2.putText(frame, "X:{}, Y:{}".format(round(puck_x,1), round(puck_y,1)), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, .8, (100,100,0), 2, cv2.LINE_AA)
		cv2.putText(frame, "Speed: {}pps".format(round(speed, 1)), (5, 125), cv2.FONT_HERSHEY_SIMPLEX, .8, (100,100,0), 1, cv2.LINE_AA)

		cv2.imshow("Croped", frame)

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