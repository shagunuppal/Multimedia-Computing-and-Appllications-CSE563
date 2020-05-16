import numpy as np
import scipy
import cv2
import os
import matplotlib.pyplot as plt
import argparse
import timeit
import json

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=float, default=2**(0.5), help="")
parser.add_argument('--bin_size', type=int, default=10, help="")
parser.add_argument('--num_distances', type=int, default=3, help="")
parser.add_argument('--threshold', type=float, default=0.5, help="")
parser.add_argument('--num_bins', type=int, default=26, help="")
parser.add_argument('--img_size', type=int, default=128, help="")
parser.add_argument('--num_quantized_colors', type=int, default=26**3, help="")
parser.add_argument('--overlapping_threshold', type=int, default=0.5, help="")

FLAGS = parser.parse_args()

def quantize_image_colors(img):
	img_quantized = np.zeros((FLAGS.img_size, FLAGS.img_size))
	
	# quantize colors of the image
	for h in range(FLAGS.img_size):
		for w in range(FLAGS.img_size):
			r, g, b = img[h, w, 0], img[h, w, 1], img[h, w, 2]
			r_bin, g_bin, b_bin = r //  FLAGS.bin_size, g // FLAGS.bin_size, b // FLAGS.bin_size
			quantized_pixel_value = r_bin + g_bin * FLAGS.num_bins + b_bin * (FLAGS.num_bins)**2
			img_quantized[h, w] = quantized_pixel_value

	# get probability matrix for quantized colors
	
	distances = [1, 2, 5]
	matrix = np.zeros((FLAGS.num_quantized_colors, len(distances)))
	c_numerators = np.zeros((FLAGS.num_quantized_colors, len(distances)))
	c_denominators = np.zeros((FLAGS.num_quantized_colors, len(distances)))

	for d in range(len(distances)):
		dist = distances[d]
		for h in range(FLAGS.img_size):
			for w in range(FLAGS.img_size):
				color = (int)(img_quantized[h, w])
				
				corner_one = [h-d,w-d] # top - left 
				corner_two = [h-d,w+d] # top - right
				corner_three = [h+d,w-d] # bottom - left
				corner_four = [h+d,w+d] # bottom - right

				if(h-d>=0): # top row
					for i in range(w-d, w+d+1):
						if (i>=0 and i<FLAGS.img_size):
							if(img_quantized[h-d][i]==color):
								c_numerators[color, d] += 1
							c_denominators[color, d] += 1
				if(h+d<FLAGS.img_size): # bottom row
					for i in range(w-d, w+d+1):
						if (i>=0 and i<FLAGS.img_size):
							if(img_quantized[h+d][i]==color):
								c_numerators[color, d] += 1
							c_denominators[color, d] += 1
				if(w-d>=0):
					for i in range(h-d, h+d+1):
						if(i>=0 and i<FLAGS.img_size):
							if(img_quantized[i][w-d]==color):
								c_numerators[color, d] += 1
							c_denominators[color, d] += 1
				if(w+d<FLAGS.img_size):
					for i in range(h-d, h+d+1):
						if(i>=0 and i<FLAGS.img_size):
							if(img_quantized[i][w+d]==color):
								c_numerators[color, d] += 1
							c_denominators[color, d] += 1
		for m in range(FLAGS.num_quantized_colors):
			if(c_denominators[m, d]!=0):
				matrix[m, d] = c_numerators[m, d] / c_denominators[m, d]
			else:
				matrix[m, d] = 0

	return img_quantized, matrix

def matching_images(img1, img2):
	
	img1_feature_vector, img2_feature_vector = img1, img2

	summation = 0
	
	for i in range(FLAGS.num_quantized_colors):
		for j in range(FLAGS.num_distances):
			img1_i_j = img1_feature_vector[i, j]
			img2_i_j = img2_feature_vector[i, j]

			numerator = abs(img1_i_j - img2_i_j)
			denominator = 1 + img1_i_j + img2_i_j

			summation += (numerator / denominator)

	summation /= FLAGS.num_quantized_colors
	return summation

def get_F1(p, r):
	return (2*p*r) / (p+r)

if __name__ == '__main__':
	
	# save feature database
	'''
	path = '../images/'
	dir_list = os.listdir(path)
	count = 0
	for i in dir_list:
		print(count)
		img_path = os.path.join(path, i)
		img_name = i.split('.')[0]
		img = cv2.imread(img_path)
		img_color = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
		img = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
		img, probability_matrix = quantize_image_colors(img)
		np.save('./stored_features_ccv/'+img_name, probability_matrix)
		count += 1'''

	# query processing
	ks = [100, 200, 350, 500]
	query_path = '../train/query/'
	dir_list = os.listdir(query_path)
	
	gt_path = '../train/ground_truth/'
	gt_dir_list = os.listdir(gt_path)

	ground_truths = ['_good', '_junk', '_ok']
	count = 0
	img_idx = 0

	precision_min = np.zeros((len(dir_list)))
	precision_max = np.zeros((len(dir_list)))
	precision_average = np.zeros((len(dir_list)))

	recall_min = np.zeros((len(dir_list)))
	recall_max = np.zeros((len(dir_list)))
	recall_average = np.zeros((len(dir_list)))

	good_average = np.zeros((len(dir_list)))
	junk_average = np.zeros((len(dir_list)))
	ok_average = np.zeros((len(dir_list)))

	time_per_query = np.zeros((len(dir_list)))

	dict_1 = {}

	for i in dir_list:

		t1 = timeit.timeit('char in text', setup='text = "sample string"; char = "g"')
		img_idx += 1
		print('Query number: ', img_idx)

		precision_over_ks = np.zeros((len(ks)))
		recall_over_ks = np.zeros((len(ks)))

		gt_good = []
		gt_junk = []
		gt_ok = []
		gt_total = []

		similarity_scores = []
		query_name = i
		count = 0
		with open(query_path+i, 'r') as f:
			r = f.readlines()[0].split()[0]
			file_name = r[5:]
			i1_feature_vector = np.load('./stored_features_ccv/'+file_name+'.npy')

			g_file_path = gt_path + query_name[:-10] +  '_good.txt'
			with open(g_file_path, 'r') as f1:
					r1 = f1.readlines()
					for line in r1:
						gt_good.append(line.rstrip())
						gt_total.append(line.rstrip())
			
			g_file_path = gt_path + query_name[:-10] +  '_junk.txt'
			with open(g_file_path, 'r') as f1:
					r1 = f1.readlines()
					for line in r1:
						gt_junk.append(line.rstrip())
						gt_total.append(line.rstrip())

			g_file_path = gt_path + query_name[:-10] +  '_ok.txt'
			with open(g_file_path, 'r') as f1:
					r1 = f1.readlines()
					for line in r1:
						gt_ok.append(line.rstrip())
						gt_total.append(line.rstrip())

			for j in os.listdir('./stored_features_ccv/'):
				if(count%1000==0):
					print('Images done', count)
				count += 1
				i2_feature_vector = np.load('./stored_features_ccv/'+j)
				score = matching_images(i1_feature_vector, i2_feature_vector)
				similarity_scores.append((score, j))
				
			similarity_scores = sorted(similarity_scores, key=lambda x: x[0])
			
			for k in range(len(ks)):
				top_k = similarity_scores[:ks[k]]
				top_k_files = [top_k[ix][1].split('.')[0] for ix in range(len(top_k))]
				
				matches, imgs = len(np.intersect1d(top_k_files, gt_total)), np.intersect1d(top_k_files, gt_total)
				matches_good = len(np.intersect1d(top_k_files, gt_good))
				matches_junk = len(np.intersect1d(top_k_files, gt_junk))
				matches_ok = len(np.intersect1d(top_k_files, gt_ok))

				dict_1[file_name] = imgs

				precision = matches / ks[k]
				recall = matches / len(gt_total)

				precision_over_ks[k] = precision
				recall_over_ks[k] = recall

				precision_min[img_idx] = np.min(precision_over_ks)
				precision_max[img_idx] = np.max(precision_over_ks)
				precision_average[img_idx] = np.mean(precision_over_ks)

				recall_min[img_idx] = np.min(recall_over_ks)
				recall_max[img_idx] = np.max(recall_over_ks)
				recall_average[img_idx] = np.mean(recall_over_ks)

				good_average[img_idx] = matches_good / matches
				junk_average[img_idx] = matches_junk / matches
				ok_average[img_idx] = matches_ok / matches

		t2 = timeit.timeit('char in text', setup='text = "sample string"; char = "g"')
		time_per_query[img_idx] = t2 - t1

	mean_precision_min, mean_recall_min = np.mean(precision_min), np.mean(recall_min)
	mean_precision_max, mean_recall_max = np.mean(precision_max), np.mean(recall_max)
	mean_precision_average, mean_recall_average = np.mean(precision_average), np.mean(recall_average)

	F1_min = get_F1(mean_precision_min, mean_recall_min)
	F1_max = get_F1(mean_precision_max, mean_recall_max)
	F1_mean = get_F1(mean_precision_average, mean_recall_average)

	good_percentage = np.mean(good_average)
	junk_percentage = np.mean(junk_average)
	ok_percentage = np.mean(ok_average)

	average_time_per_query = np.mean(time_per_query)

	print('Mean precision minimum: ', mean_precision_min)
	print('Mean precision maxinum: ', mean_precision_max)
	print('Mean precision average: ', mean_precision_average)

	print('Mean recall minimum: ', mean_recall_min)
	print('Mean recall maximum: ', mean_recall_max)
	print('Mean recall average: ', mean_recall_average)

	print('Mean F1 minimum: ', F1_min)
	print('Mean F1 maximum: ', F1_max)
	print('Mean F1 average: ', F1_mean)

	print('good percentage: ', good_percentage)
	print('junk percentage: ', junk_percentage)
	print('ok percentage:', ok_percentage)
	
	print('Average time per query: ', average_time_per_query)

	with open('./color_correlogram_matches.json', 'w') as f:
		json.dump(dict_1, f)