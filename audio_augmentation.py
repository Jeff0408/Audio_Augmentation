import librosa
# import matplotlib.pyplot as plt
import numpy as np
import random
import os , time

# n_mfcc = 39
# numbers of augmentation
augmentation_num = 10
# sample parameters
sample_rate = 16000
# range of pitch
pitch_shift1, pitch_shift2 = 0.01 , 5.0
# range of pitch time strentch
time_stretch1, time_stretch2 = 0.05, 0.25
# range of noise
noise1, noise2 = 0.05, 0.25
# range of time shift
shift_max1, shift_max2 = 10, 30

def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift_time(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data



def audio_augmentation(wav_file):

    y, sr = librosa.load(wav_file, sr=sample_rate)

    for j in range(augmentation_num):
        rd1 = random.uniform(pitch_shift1, pitch_shift2)
        ii = random.choice((-1, 1))
        rd2 = random.uniform(time_stretch1, time_stretch2)
        rd2 = 1.0 + ii * rd2
        rd3 = random.uniform(noise1, noise2)
        rd4 = random.uniform(shift_max1, shift_max2)

        y_ps = librosa.effects.pitch_shift(y, sr, n_steps = rd1)
        y_ts = librosa.effects.time_stretch(y, rate = rd2)
        y_ns = add_noise(y, noise_factor = rd3)
        y_ss = shift_time(y, sample_rate, rd4, 'right')

        tmp1 = wav_file.split('/')
        tmp2 = tmp1[-1].split('.')
        wav_name = tmp2[0]
        tmp_path1 = save_audio_path + wav_name[0:2] + '/'
        if not os.path.exists(tmp_path1):
            os.mkdir(tmp_path1)

        ps_wav = tmp_path1 + wav_name + '_ps_' + str(j) + '.wav'
        ts_wav = tmp_path1 + wav_name + '_ts_' + str(j) + '.wav'
        ns_wav = tmp_path1 + wav_name + '_ns_' + str(j) + '.wav'
        ss_wav = tmp_path1 + wav_name + '_ss_' + str(j) + '.wav'


        librosa.output.write_wav(ps_wav, y_ps, sample_rate)
        librosa.output.write_wav(ts_wav, y_ts, sample_rate)
        librosa.output.write_wav(ns_wav, y_ns, sample_rate)
        librosa.output.write_wav(ss_wav, y_ss, sample_rate)

    return

def filter_file(path):
    flist = os.listdir(path)
    all_file = []
    for filename in flist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            all_file.extend(filter_file(filepath))
        else:
            all_file.append(filepath)
    return all_file

if __name__ == "__main__":

    audio_path = '/home/jeff/Desktop/Sound/wav16/'
    save_audio_path = '/home/jeff/Desktop/Sound/wav16/res/'

    all_file = filter_file(audio_path)
    random.shuffle(all_file)
    file_len = len(all_file)
    print('wav file len:',file_len)

    print('start wav data augmentation ...')

    for i in range(file_len):
        audio_file = all_file[i]
        audio_augmentation(audio_file)

        if (i % 100 == 0):
            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
            print('time:', now_time)
            print('predict num:', i)

    print('wav data augmentation done ...')

# audio1 = 'test.wav'
# y1, sr1 = librosa.load(audio1, sr=sr)
# mfccs1 = np.array(librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=n_mfcc).T)
# y_shape1 = mfccs1.shape[0]

# plt.subplot(311)
# plt.plot(y)
# plt.title('Original waveform')
# plt.axis([0, 200000, -0.4, 0.4])
# plt.subplot(312)
# plt.plot(y_ps)
# plt.title('Pitch Shift transformed waveform')
# plt.axis([0, 200000, -0.4, 0.4])
# plt.subplot(313)
# plt.plot(y_ts)
# plt.title('Time Stretch transformed waveform')
# plt.axis([0, 200000, -0.4, 0.4])
# plt.tight_layout()
# plt.show()
