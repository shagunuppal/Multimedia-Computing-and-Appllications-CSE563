import numpy as np 
import cv2
import os
from skimage.feature import hessian_matrix, hessian_matrix_det, peak_local_max
from skimage.transform import integral_image
import argparse
import math
import matplotlib.pyplot as plt
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


def find_feature_vector(image):
	# Integral Images
	feature_vector = []
	integrated_img = integral_image(image)
	S = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

	for s in S:
		hess_matrix = hessian_matrix_det(image, sigma=s)
		potential_keypoints = peak_local_max(hess_matrix, num_peaks=50)
		
		for p in potential_keypoints:
			point = [(float)(s), (float)(p[0]), (float)(p[1])]
			feature_vector.append(point)

	refined_keypoints = remove_overlapping_blobs(feature_vector)
	return list(feature_vector)

if __name__ == '__main__':
	path = '../images/'
	dir_list = os.listdir(path)
	count = 0

	json_dict = {}

	for i in dir_list:
		i_name = i.split('.')[0]

		img_path = os.path.join(path, i)
		img = cv2.imread(img_path)
		img_color = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
		img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		
		key_points_refined = find_feature_vector(img)
		json_dict[i_name] = key_points_refined

		# fig, ax = plt.subplots()
		# nh,nw = FLAGS.img_size, FLAGS.img_size

		# ax.imshow(img_color, interpolation='nearest', cmap="gray")
		# for blob in key_points_refined:
		#     sigma, x, y = blob
		#     c = plt.Circle((y, x), sigma , color='red', linewidth=1.5, fill=False)
		#     ax.add_patch(c)

		# ax.plot() 
		# plt.savefig('./surf_images/'+(str)(count)+'.png')
		# plt.close()

		count += 1
		print(count)

	with open('./surf_keypoints.json', 'w') as f:
			json.dump(json_dict, f)