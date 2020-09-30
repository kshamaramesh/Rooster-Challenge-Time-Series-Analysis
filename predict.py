import soundfile as sf
f = sf.SoundFile('rooster_competition.wav')
print('samples = {}'.format(len(f)))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(len(f) / f.samplerate))


# In[89]:


from IPython.display import Audio
from scipy.io import wavfile

samplerate, tabulasa = wavfile.read('rooster_competition.wav')

start = samplerate * 5 # 10 seconds in
end = start + samplerate * 2 # 5 second duration
Audio(data=tabulasa[start:end, 0], rate=samplerate)


# In[90]:


from sklearn.decomposition import PCA
import numpy as np

def pca_reduce(signal, n_components, block_size=1024):
    
    # First, zero-pad the signal so that it is divisible by the block_size
    samples = len(signal)
    hanging = block_size - np.mod(samples, block_size)
    padded = np.lib.pad(signal, (0, hanging), 'constant', constant_values=0)
    
    # Reshape the signal to have 1024 dimensions
    reshaped = padded.reshape((len(padded) // block_size, block_size))
    
    # Second, do the actual PCA process
    pca = PCA(n_components=n_components)
    pca.fit(reshaped)
    
    transformed = pca.transform(reshaped)
    reconstructed = pca.inverse_transform(transformed).reshape((len(padded)))
    return pca, transformed, reconstructed


# In[91]:


tabulasa_left = tabulasa[:,0]

_, _, reconstructed = pca_reduce(tabulasa_left, 1024, 1024)

Audio(data=reconstructed[start:end], rate=samplerate)


# In[92]:


_, _, reconstructed = pca_reduce(tabulasa_left, 32, 1024)

Audio(data=reconstructed[start:end], rate=samplerate)


# In[93]:


from bz2 import compress
import pandas as pd

def raw_estimate(transformed, pca):
    # We assume that we'll be storing things as 16-bit WAV,
    # meaning two bytes per sample
    signal_bytes = transformed.tobytes()
    # PCA stores the components as floating point, we'll assume
    # that means 32-bit floats, so 4 bytes per element
    component_bytes = transformed.tobytes()
    
    # Return a result in megabytes
    return (len(signal_bytes) + len(component_bytes)) / (2**20)

# Do an estimate for lossless compression applied on top of our
# PCA reduction
def bz2_estimate(transformed, pca):
    bytestring = transformed.tobytes() + b';' + pca.components_.tobytes()
    compressed = compress(bytestring)
    return len(compressed) / (2**20)

compression_attempts = [
    (1, 1),
    (1, 2),
    (1, 4),
    (4, 32),
    (16, 256),
    (32, 256),
    (64, 256),
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (128, 2048),
    (256, 2048),
    (512, 2048),
    (1024, 2048)
]

def build_estimates(signal, n_components, block_size):
    pca, transformed, recon = pca_reduce(tabulasa_left, n_components, block_size)
    raw_pca_estimate = raw_estimate(transformed, pca)
    bz2_pca_estimate = bz2_estimate(transformed, pca)
    raw_size = len(recon.tobytes()) / (2**20)
    return raw_size, raw_pca_estimate, bz2_pca_estimate

pca_compression_results = pd.DataFrame([
        build_estimates(tabulasa_left, n, bs)
        for n, bs in compression_attempts
    ])

pca_compression_results.columns = ["Raw", "PCA", "PCA w/ BZ2"]
pca_compression_results.index = compression_attempts
pca_compression_results


# In[94]:


_, _, reconstructed = pca_reduce(tabulasa_left, 16, 256)
Audio(data=reconstructed[start:end], rate=samplerate)


# In[95]:


_, _, reconstructed = pca_reduce(tabulasa_left, 1, 4)
Audio(data=reconstructed[start:end], rate=samplerate)


# In[96]:


_, _, reconstructed = pca_reduce(tabulasa_left, 64, 256)
Audio(data=reconstructed[start:end], rate=samplerate)


# In[98]:


import librosa
import noisereduce as nr
# crows cawing
fn = "rooster_competition.wav"
audio_data, sampling_rate = librosa.load(fn)
# sound of wind blowing
noisy_part = audio_data[8000:10000]
# perform noise reduction
reduced_noise = nr.reduce_noise(audio_clip=audio_data,
noise_clip=noisy_part, verbose=True)


# In[99]:


import librosa
trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)


# In[107]:


index


# In[106]:


trimmed


# In[112]:

import sys
from aubio import source, pitch

win_s = 4096
hop_s = 512 

s = source('rooster_competition.wav', samplerate, hop_s)
samplerate = s.samplerate

tolerance = 0.8

pitch_o = pitch("yin", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches = []
confidences = []

total_frames = 0
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    pitches += [pitch]
    confidence = pitch_o.get_confidence()
    confidences += [confidence]
    total_frames += read
    if read < hop_s: break

print("max frequency = " + str(np.array(pitches).max()) + " hz")



import numpy as np

def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    result_d = {
        'mean': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt
    }

    return result_d



import librosa
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

np.set_printoptions(threshold=sys.maxsize)

filename = 'rooster_competition.wav'
Fs = 44100
clip, sample_rate = librosa.load(filename, sr=Fs)

n_fft = 1024  # frame length 
start = 0 

hop_length=512

#commented out code to display Spectrogram
X = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)

#now print all values 

t_samples = np.arange(clip.shape[0]) / Fs
t_frames = np.arange(X.shape[1]) * hop_length / Fs
#f_hertz = np.arange(N / 2 + 1) * Fs / N       # Works only when N is even
f_hertz = np.fft.rfftfreq(n_fft, 1 / Fs)         # Works also when N is odd

#example
print('Time (seconds) of last sample:', t_samples[-1])
print('Time (seconds) of last frame: ', t_frames[-1])
print('Frequency (Hz) of last bin:   ', f_hertz[-1])

print('Time (seconds) :', len(t_samples))

#prints array of time frames 
print('Time of frames (seconds) : ', t_frames)
#prints array of frequency bins
print('Frequency (Hz) : ', f_hertz)

print('Number of frames : ', len(t_frames))
print('Number of bins : ', len(f_hertz))

#This code is working to printout frame by frame intensity of each frequency
#on top line gives freq bins
curLine = 'Bins,'
for b in range(1, len(f_hertz)):
    curLine += str(f_hertz[b]) + ','
print(curLine)

curLine = ''
for f in range(1, len(t_frames)):
    curLine = str(t_frames[f]) + ','
    for b in range(1, len(f_hertz)): #for each frame, we get list of bin values printed
        curLine += str("%.02f" % np.abs(X[b, f])) + ','
        #remove format of the float for full details if needed
        #curLine += str(np.abs(X[b, f])) + ','
        #print other useful info like phase of frequency bin b at frame f.
        #curLine += str("%.02f" % np.angle(X[b, f])) + ',' 
    print(curLine)



[j-i for i, j in zip(f_hertz[:-1], f_hertz[1:])]



from numpy import array, diff, where, split
from scipy import arange
import soundfile
import numpy, scipy
import pylab
import copy
import matplotlib
matplotlib.use('tkagg')

def findPeak(magnitude_values, noise_level=2000):
    
    splitter = 0
    # zero out low values in the magnitude array to remove noise (if any)
    magnitude_values = numpy.asarray(magnitude_values)        
    low_values_indices = magnitude_values < noise_level  # Where values are low
    magnitude_values[low_values_indices] = 0  # All low values will be zero out
    
    indices = []
    
    flag_start_looking = False
    
    both_ends_indices = []
    
    length = len(magnitude_values)
    for i in range(length):
        if magnitude_values[i] != splitter:
            if not flag_start_looking:
                flag_start_looking = True
                both_ends_indices = [0, 0]
                both_ends_indices[0] = i
        else:
            if flag_start_looking:
                flag_start_looking = False
                both_ends_indices[1] = i
                # add both_ends_indices in to indices
                indices.append(both_ends_indices)
                
    return indices

def extractFrequency(indices, freq_threshold=2):
    
    extracted_freqs = []
    
    for index in indices:
        freqs_range = freq_bins[index[0]: index[1]]
        avg_freq = round(numpy.average(freqs_range))
        
        if avg_freq not in extracted_freqs:
            extracted_freqs.append(avg_freq)

    # group extracted frequency by nearby=freq_threshold (tolerate gaps=freq_threshold)
    group_similar_values = split(extracted_freqs, where(diff(extracted_freqs) > freq_threshold)[0]+1 )
    
    # calculate the average of similar value
    extracted_freqs = []
    for group in group_similar_values:
        extracted_freqs.append(round(numpy.average(group)))
    
    print("freq_components", extracted_freqs)
    return extracted_freqs

if __name__ == '__main__':
    
    file_path = 'rooster_competition.wav'
    print('Open audio file path:', file_path)
    
    audio_samples, sample_rate  = soundfile.read(file_path, dtype='int16')
    number_samples = len(audio_samples)
    print('Audio Samples: ', audio_samples)
    print('Number of Sample', number_samples)
    print('Sample Rate: ', sample_rate)
    
    # duration of the audio file
    duration = round(number_samples/sample_rate, 2)
    print('Audio Duration: {0}s'.format(duration))
    
    # list of possible frequencies bins
    freq_bins = arange(number_samples) * sample_rate/number_samples
    print('Frequency Length: ', len(freq_bins))
    print('Frequency bins: ', freq_bins)
    
#     # FFT calculation
    fft_data = scipy.fft(audio_samples)
    print('FFT Length: ', len(fft_data))
    print('FFT data: ', fft_data)

    freq_bins = freq_bins[range(number_samples//2)]      
    normalization_data = fft_data/number_samples
    magnitude_values = normalization_data[range(len(fft_data)//2)]
    magnitude_values = numpy.abs(magnitude_values)
        
    indices = findPeak(magnitude_values=magnitude_values, noise_level=200)
    frequencies = extractFrequency(indices=indices)
    print("frequencies:", frequencies)
    
    x_asis_data = freq_bins
    y_asis_data = magnitude_values
 
    pylab.plot(x_asis_data, y_asis_data, color='blue') # plotting the spectrum
  
    pylab.xlabel('Freq (Hz)')
    pylab.ylabel('|Magnitude - Voltage  Gain / Loss|')
    pylab.show()


def audio_player_list(signals, rates, width=270, height=40, columns=None, column_align='center'):
    """Generate list of audio players

    Notebook: B/B_PythonAudio.ipynb

    Args:
        signals: List of audio signals
        rates: List of sample rates
        width: Width of player (either number or list)
        height: Height of player (either number or list)
        columns: Column headings
        column_align: Left, center, right
    """
    pd.set_option('display.max_colwidth', None)

    if isinstance(width, int):
        width = [width] * len(signals)
    if isinstance(height, int):
        height = [height] * len(signals)

    audio_list = []
    for cur_x, cur_Fs, cur_width, cur_height in zip(signals, rates, width, height):
        audio_html = ipd.Audio(data=cur_x, rate=cur_Fs)._repr_html_()
        audio_html = audio_html.replace('\n', '').strip()
        audio_html = audio_html.replace('<audio ', f'<audio style="width: {cur_width}px; height: {cur_height}px" ')
        audio_list.append([audio_html])

    df = pd.DataFrame(audio_list, index=columns).T
    table_html = df.to_html(escape=False, index=False, header=bool(columns))
    table_html = table_html.replace('<th>', f'<th style="text-align: {column_align}">')
    ipd.display(ipd.HTML(table_html))
    
fn_wav = os.path.join('rooster_competition.wav')
x, Fs = librosa.load(fn_wav, sr=None)

audio_player_list([x, x, x, x], [Fs, Fs, Fs, Fs], width=120, height=20, 
                  columns=['a', 'b', 'c', 'd'])

audio_player_list([x, x, x], [Fs, Fs, Fs], width=200, height=80, 
                  columns=['a', 'b', 'c'], column_align='left')


audio_player_list([x, x, x, x], [Fs, Fs, Fs, Fs], 
                  width=[40, 80, 150, 300], height=[20, 40, 60, 80], 
                  columns=['a', 'b', 'c', 'd'], column_align='right')


#predict the model
y=model.predict(audio_player_list)
if np.max(y)>=0.9:
                                print ( np.max(y))
                                print (int_to_word_out[np.argmax(y)])
                                results.append(int_to_word_out[np.argmax(y)])
                                mylist = list(dict.fromkeys(results))


               
else:
                   
                                results.append('norooster')
                               
                                mylist = list(dict.fromkeys(results))
row_id = list(prediction_dict.keys())
rooster = list(prediction_dict.values())

prediction_df = pd.DataFrame({
    "roostername": rooster_name,
    "rooster": rooster
    "crowing":crowing
})
prediction_df.to_csv("submission.csv", index=False)













