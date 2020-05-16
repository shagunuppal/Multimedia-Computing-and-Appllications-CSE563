import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io.wavfile
from scipy.fftpack import dct as DCT
import random

train_folder = './Dataset/training'
validation_folder = './Dataset/validation'

is_train = False
add_noise = False

train_classes = os.listdir(train_folder)
val_classes = os.listdir(validation_folder)

if(add_noise==False):
	if (not os.path.exists('./saved_features_mfcc')):
		os.mkdir('./saved_features_mfcc')

	if(is_train==True):
		classes = train_classes
		folder = train_folder
		for t in train_classes:
			if not os.path.exists('./saved_features_mfcc/train/'+t):
				os.mkdir('./saved_features_mfcc/train/'+t)
	elif(is_train==False):
		classes = val_classes
		folder = validation_folder
		for t in val_classes:
			if not os.path.exists('./saved_features_mfcc/val/'+t):
				os.mkdir('./saved_features_mfcc/val/'+t)

if(add_noise==True):
	if (not os.path.exists('./saved_features_mfcc_noise')):
		os.mkdir('./saved_features_mfcc_noise')

	if(is_train==True):
		classes = train_classes
		folder = train_folder
		for t in train_classes:
			if not os.path.exists('./saved_features_mfcc_noise/train/'+t):
				os.mkdir('./saved_features_mfcc_noise/train/'+t)
	elif(is_train==False):
		classes = val_classes
		folder = validation_folder
		for t in val_classes:
			if not os.path.exists('./saved_features_mfcc_noise/val/'+t):
				os.mkdir('./saved_features_mfcc_noise/val/'+t)


emphasis_ratio = 0.95
frame_size = 0.025
frame_overlap = 0.010
nfft = 512
num_filters = 40
n_cepstral = 12
cep_lifter = 10

def plot_raw_signal(signal, file_name):
	plt.figure(1)
	plt.title("Signal Wave")
	plt.plot(signal)
	plt.savefig('./raw_signals/'+file_name+'.png')
	plt.close()

def get_Hz_scale_vec(ks,sample_rate,Npoints):
	freq_Hz = ks*sample_rate/Npoints
	freq_Hz  = [int(i) for i in freq_Hz ] 
	return(freq_Hz )

def plot_mfcc_features(signal, file_name, sample_rate, ts, L=256, mappable = None):
	plt_signal = plt.imshow(signal, origin='lower')
	## create ylim
	Nyticks = 10
	ks = np.linspace(0, signal.shape[0], Nyticks)
	ksHz = get_Hz_scale_vec(ks, sample_rate, ts)
	plt.yticks(ks,ksHz)
	plt.ylabel("Frequency (Hz)")

	## create xlim
	Nxticks = 10
	plt.xlabel("Time (sec)")

	#plt.title("Spectrogram L={} Spectrogram.shape={}".format(L, signal.shape))
	plt.colorbar(mappable, use_gridspec=True)
	plt.savefig(file_name+'.png')

def compute_mfcc_features(audio_signal, sampling_rate, t, file):
	audio_signal_emphasised = np.append(audio_signal[0], audio_signal[1:] - emphasis_ratio * audio_signal[:-1])
	signal_length = len(audio_signal_emphasised)
	
	frame_step = (int)(round(frame_overlap * sampling_rate))
	frame_len = (int)(round(frame_size * sampling_rate))
	frames_num = (int)(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step))

	signal_padding_length = frame_step * frames_num + frame_len
	padding_value = np.zeros(signal_padding_length - signal_length)
	padded_signal = np.append(audio_signal_emphasised, padding_value)

	a = np.tile(np.arange(0, frame_len), (frames_num, 1))
	b = np.tile(np.arange(0, frames_num * frame_step, frame_step), (frame_len, 1)).T
	index = a + b
	index = index.astype(np.int32, copy=False)
	frames = padded_signal[index]
	frames = frames * np.hamming(frame_len)

	frames_magnitude = np.absolute(np.fft.rfft(frames, nfft))
	frames_power = (1. / nfft) * (frames_magnitude ** 2)

	mel_frequency_low = 0
	mel_frequency_high = np.log10(1 + sampling_rate / 1400) * 2595

	mel_freq_points = np.linspace(mel_frequency_low, mel_frequency_high, num_filters + 2)
	hz_freq_points = 700 * (10 ** (mel_freq_points / 2595)  -1)
	freq_bins = np.floor((nfft + 1) * hz_freq_points / sampling_rate)

	freq_b = np.zeros((num_filters, nfft // 2 + 1))
	for f in range(1, num_filters + 1):
		f_m_prev = int(freq_bins[f - 1])
		f_m = int(freq_bins[f])
		f_m_next = int(freq_bins[f + 1])

		for f1 in range(f_m_prev, f_m):
			freq_b[f - 1, f1] = (f1 - freq_bins[f - 1]) / (freq_bins[f] - freq_bins[f - 1])
		for f1 in range(f_m, f_m_next):
			freq_b[f - 1, f1] = (freq_bins[f + 1] - f1) / (freq_bins[f + 1] - freq_bins[f])

	filter_b = np.dot(frames_power, freq_b.T)
	filter_b = np.where(filter_b == 0, np.finfo(float).eps, filter_b)
	filter_b = np.log10(filter_b) * 20

	mfcc_features = DCT(filter_b)[:, 1:(n_cepstral + 1)]
	nframes = mfcc_features.shape[0]
	ncoeffs = mfcc_features.shape[1]

	N = np.arange(ncoeffs)
	mfcc_features  = mfcc_features * (1 + (cep_lifter / 2) * np.sin(np.pi * N / cep_lifter))

	filter_b-= (np.mean(filter_b, axis=0) + 1e-8)
	mfcc_features = mfcc_features - (np.mean(mfcc_features, axis=0) + 1e-8)

	print(mfcc_features.shape)

	#plot_raw_signal(mfcc_features, t+'_'+file+'_mfcc_features')
	#plot_mfcc_features(mfcc_features, './mfcc_features', sampling_rate, signal_length)
	if(is_train):
		save_path = './saved_features_mfcc/train/'+t+'/'+file
	else:
		save_path = './saved_features_mfcc/val/'+t+'/'+file
	print(save_path)

	np.save(save_path, mfcc_features)

	return mfcc_features, signal_length

if(add_noise==True):
	noise_signals = []
	eps = 0.001
	noise_folder = './Dataset/_background_noise_'
	noise_files = os.listdir(noise_folder)
	for f in noise_files:
		file_path = os.path.join(noise_folder, f)
		noise_sampling_rate, noise_audio_signal = scipy.io.wavfile.read(file_path)
		noise_signals.append((noise_sampling_rate, noise_audio_signal * eps))
	
for t in classes:
	data_files = os.listdir(os.path.join(folder, t))
	for file in data_files:
		
		file_name = os.path.join(folder, t, file)
		sampling_rate, audio_signal = scipy.io.wavfile.read(file_name)

		pad_seq = np.zeros(abs(sampling_rate - len(audio_signal)))
		audio_signal = np.append(audio_signal, pad_seq)
		#plot_raw_signal(audio_signal, t+'_'+file)

		mfcc_features, sig_len = compute_mfcc_features(audio_signal, sampling_rate, t, file)
		
		if(add_noise):
			noise_idx = random.randint(0, len(noise_signals) - 1)
			noise_sr, noise_sig = noise_signals[noise_idx]
			ix = random.randint(0, noise_sig.shape[0] - sampling_rate - 1)
			noise_sig = noise_sig[ix : ix + sampling_rate]
			audio_signal += noise_sig
			mfcc_features, sig_len = compute_mfcc_features(audio_signal, sampling_rate, t, file + '_noise')
		
# reference to understand mffc feature computation: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
