import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import random

train_folder = './Dataset/training'
validation_folder = './Dataset/validation'

is_train = False
add_noise = True

train_classes = os.listdir(train_folder)
val_classes = os.listdir(validation_folder)

nfft = 512

if(add_noise==False):
	if (not os.path.exists('./saved_features_spectrogram')):
		os.mkdir('./saved_features_spectrogram')

	if(is_train==True):
		classes = train_classes
		folder = train_folder
		for t in train_classes:
			if not os.path.exists('./saved_features_spectrogram/train/'+t):
				os.mkdir('./saved_features_spectrogram/train/'+t)
	else:
		classes = val_classes
		folder = validation_folder
		for t in val_classes:
			if not os.path.exists('./saved_features_spectrogram/val/'+t):
				os.mkdir('./saved_features_spectrogram/val/'+t)
elif(add_noise==True):
	if (not os.path.exists('./saved_features_spectrogram_noise')):
		os.mkdir('./saved_features_spectrogram_noise')

	if(is_train==True):
		classes = train_classes
		folder = train_folder
		for t in train_classes:
			if not os.path.exists('./saved_features_spectrogram_noise/train/'+t):
				os.mkdir('./saved_features_spectrogram_noise/train/'+t)
	else:
		classes = val_classes
		folder = validation_folder
		for t in val_classes:
			if not os.path.exists('./saved_features_spectrogram_noise/val/'+t):
				os.mkdir('./saved_features_spectrogram_noise/val/'+t)


def get_Hz_scale_vec(ks,sample_rate,Npoints):
	freq_Hz = ks * sample_rate / Npoints
	freq_Hz  = [int(i) for i in freq_Hz ] 
	return(freq_Hz )

def plot_signal_in_time_domain(signal, sampling_rate, file_name):
	signal_duration = len(signal) / sampling_rate
	plt.figure()
	plt.plot(ts)
	plt.xticks(np.arange(0, len(signal), sampling_rate), np.arange(0, len(signal) / sampling_rate, 1))
	plt.xlabel("Time (in seconds)")
	plt.ylabel("Amplitude")
	plt.savefig(file_name+'.png')

def get_ms_breakdown(sig, ind):
	x = np.arange(0, len(sig))
	b = np.sum(sig * np.exp((1j * 2 * np.pi * x * ind) / len(sig))) / len(sig)
	#b = np.sum(sig * np.exp((1j * 2 * np.pi * x * ind) / len(sig))) / len(sig)
	return b

def plot_raw_signal(signal, file_name):
	plt.figure(1)
	plt.title("Signal Wave")
	plt.plot(signal)
	plt.savefig('./raw_signals/'+file_name+'.png')
	plt.close()

def get_ms(sig):
	arr = []
	L = len(sig)
	for i in range(L // 2):
		p = get_ms_breakdown(sig, i)
		p = np.abs(p) * 2
		arr.append(p)
	return arr

def spectogram_features(signal, nfft=nfft):
	ms = []
	
	s_time = np.arange(0, len(signal), nfft, dtype=int)
	s_time = s_time[s_time + nfft < len(signal)]
	
	for s in s_time:
		wind = get_ms(signal[s : s + nfft])
		ms.append(wind)
	
	sig = np.asarray(ms).T
	ms_spec = 20 * np.log10(sig, where=sig>0)
	
	return len(signal), ms_spec


def plot_spec_features(signal, file_name, sample_rate, ts, L=256, mappable = None):
	plt_signal = plt.imshow(signal, origin='lower')
	
	Nyticks = 10
	ks      = np.linspace(0, signal.shape[0], Nyticks)
	ksHz    = get_Hz_scale_vec(ks, sample_rate, ts)
	plt.yticks(ks,ksHz)
	plt.ylabel("Frequency (Hz)")

	Nxticks = 10
	plt.xlabel("Time (sec)")

	plt.colorbar(mappable, use_gridspec=True)
	plt.savefig(file_name+'.png')

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

		pad_seq = np.zeros(sampling_rate - len(audio_signal))
		audio_signal = np.append(audio_signal, pad_seq)
		signal_length = len(audio_signal)

		if(is_train):
			save_path = './saved_features_spectrogram_noise/train/'+t+'/'+file
		else:
			save_path = './saved_features_spectrogram_noise/val/'+t+'/'+file
		print(save_path)

		s_time, spec_features = spectogram_features(audio_signal)
		np.save(save_path, spec_features)
		#plot_spec_features(spec_features, './spec_features', sampling_rate, signal_length)

		if(add_noise):
			noise_idx = random.randint(0, len(noise_signals) - 1)
			noise_sr, noise_sig = noise_signals[noise_idx]
			ix = random.randint(0, noise_sig.shape[0] - sampling_rate)
			noise_sig = noise_sig[ix : ix + sampling_rate]
			audio_signal += noise_sig
			s_time, spec_features_noise = spectogram_features(audio_signal)

			if(is_train):
				save_path = './saved_features_spectrogram_noise/train/' + t + '/ '+ file + '_noise'
			else:
				save_path = './saved_features_spectrogram_noise/val/' + t + '/' + file + '_noise'
			print(save_path)

			np.save(save_path, spec_features_noise)

# reference to understand spectrogram: https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html
