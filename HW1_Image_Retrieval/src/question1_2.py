import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--k', type=float, default=1.414, help="")
parser.add_argument('--filter_size', type=int, default=8, help="")
parser.add_argument('--sigma', type=float, default=1, help="")
parser.add_argument('--threshold', type=float, default=0.5, help="")
parser.add_argument('--num_scales', type=int, default=8, help="")
parser.add_argument('--img_size', type=int, default=128, help="")
parser.add_argument('--overlapping_threshold', type=int, default=0.5, help="")

FLAGS = parser.parse_args()

def laplacian_of_gaussian(sigma):
	filter_size = (int)(sigma*5)
	array = np.zeros((filter_size, filter_size))
	patchx = np.arange((-filter_size+1)/2, (filter_size+1)/2)
	for row in range(len(patchx)):
		for column in range(len(patchx)):
			x_inp = row
			y_inp = column
			constant = 1. / (2*np.pi*(np.power(sigma, 4)))
			p = (x_inp*x_inp + y_inp*y_inp)/(2*sigma**2)
			log = constant*(x_inp*x_inp + y_inp*y_inp -2*sigma*sigma)*np.exp(-p)
			array[row][column] = log
	return array

def conv_image_log(image):
	# use multiple sigma for different scale-normalized filters
	convolved_images = np.zeros((FLAGS.num_scales, FLAGS.img_size, FLAGS.img_size))
	sigma1 = FLAGS.sigma
	
	for i in range(0, FLAGS.num_scales):
		sigma = sigma1 * (FLAGS.k+i)
		log_filter = laplacian_of_gaussian(sigma)
		img = cv2.filter2D(image,-1, log_filter)
		img = np.square(img)
		convolved_images[i, :, :] = img
	return convolved_images

def non_max_suppression(log_convolved):
	h, w = FLAGS.img_size, FLAGS.img_size
	
	key_extrema_points = []
	for scale in range(1, FLAGS.num_scales-1):
		curr_img = log_convolved[scale, :, :]
		top_img = log_convolved[scale-1, :, :]
		bottom_img = log_convolved[scale+1, :, :]
		neighbouring_positions = [[-1, 0], [1, 0], [-1, -1], [0, -1], [-1, 1], [1, -1], [1, 0], [1, 1]]
		neighbours = []
		
		for row in range(1, h-1):
			for column in range(1, w-1):
				neighbours = []
				
				# neighbours in same plane
				for n in neighbouring_positions:
					pos_x, pos_y = row+n[0], column+n[1]
					neighbours.append(curr_img[pos_x][pos_y])
				
				# neighbous in plane above
				for n in neighbouring_positions:
					pos_x, pos_y = row+n[0], column+n[1] 
					neighbours.append(top_img[pos_x][pos_y])
				neighbours.append(top_img[pos_x][pos_y])
				
				# neighbous in plane below
				for n in neighbouring_positions:
					pos_x, pos_y = row+n[0], column+n[1] 
					neighbours.append(bottom_img[pos_x][pos_y])
				neighbours.append(bottom_img[pos_x][pos_y])
				
				if(curr_img[row][column]>max(neighbours)):
					if(curr_img[row][column]>=0.03): # removing low-contrast keypoints
						if([scale, row, column] not in key_extrema_points):
							key_extrema_points.append([scale, row, column])
	return key_extrema_points

def remove_overlapping_blobs(key_points):
	
	#print('Before overleap removal: ', len(key_points))
	k_scale = FLAGS.k
	
	count = 0
	for i in key_points:
		
		s1, x1, y1 = i[0], i[1], i[2]
		r1 = s1 * k_scale

		for j in key_points:
			if(i!=j and i[0]!=0 and j[0]!=0):
				
				s2, x2, y2 = j[0], j[1], j[2]
				r2 = s2 * k_scale

				d = math.sqrt((x2-x1)**2 + (y2-y1)**2)

				if(d <= abs(r2 - r1)): 		# one inside the other
					if(s1>s2):
						j[0] = 0
						count += 1
					else:
						i[0] = 0
						count += 1
				elif(d<r1+r2 and d>abs(r2-r1)):
					overlap_r1, overlap_r2 = (r1*r1 - r2*r2 + d*d) / (2 * r1 * d), (r2*r2 - r1*r1 + d*d) / (2 * r2 * d)
					
					if(overlap_r1>=1):
						overlap_r1 = 1
					elif(overlap_r1<=-1):
						overlap_r1 = 1

					if(overlap_r2>=1):
						overlap_r2 = 1
					elif(overlap_r2<=-1):
						overlap_r2 = -1
					
					overlap_r1 = math.acos(overlap_r1)
					overlap_r2 = math.acos(overlap_r2)

					
					q1, q2, q3, q4 = r1+r2-d, -r2+r1+d, -r1+r2+d, r1+r2+d
					
					overlap_region = (np.power(r1, 2) * overlap_r1) + (np.power(r2, 2) * overlap_r2) - (np.power(abs(q1*q2*q3*q4), 0.5))/2.0
					overlap_region /= (math.pi * (min(r1, r2) ** 2))

					if(overlap_region > FLAGS.overlapping_threshold):
						if(s1>s2):
							j[0] = 0
							count += 1
						else:
							i[0] = 0
							count += 1
					
	refined_blobs = []
	for x in key_points:
		if(x[0] > 0):
			refined_blobs.append(x)
	return refined_blobs

if __name__ == '__main__':
	path = '../images/'
	dir_list = os.listdir(path)
	count = 0

	json_dict = {}

	for i in dir_list:
		img_path = os.path.join(path, i)
		i_name = i.split('.')[0]

		img = cv2.imread(img_path)
		img_color = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
		img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		convolved_images = conv_image_log(img)
		key_points = non_max_suppression(convolved_images)
		key_points_refined = remove_overlapping_blobs(key_points)
		
		json_dict[i_name] = key_points_refined

		# fig, ax = plt.subplots()
		# nh,nw = FLAGS.img_size, FLAGS.img_size

		#################################################################################
		# ax.imshow(img_color, interpolation='nearest', cmap="gray")
		# for blob in key_points_refined:
		# 	sigma, x, y = blob
		# 	c = plt.Circle((y, x), sigma , color='red', linewidth=1.5, fill=False)
		# 	ax.add_patch(c)
		# ax.plot() 
		# plt.savefig('./blob_detection_images/'+(str)(count)+'.png')
		# plt.close()
		#################################################################################

		count += 1
		print(count)
	
	with open('./scale_invariant_log_keypoints.json', 'w') as f:
		json.dump(json_dict, f)