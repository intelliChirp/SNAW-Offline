from scipy import signal
from pyAudioAnalysis import audioSegmentation as aS
from scipy import interpolate
from scipy.signal import butter, lfilter, freqz
from scipy.stats import itemfreq
import csv
import yaml
import sys
import wave
import contextlib
import numpy as np
import librosa
import peakutils
import SimpleITK as sitk
import os
import json
import argparse
import traceback
import warnings

#################### CONSTANTS ####################

# Display All Steps - Prints when each step starts and completes
# Prediction Verbose - Prints values for each prediction
DISPLAY_ALL_STEPS = False
PREDICTION_VERBOSE = False

# Classification Threshold - Label the current timestep's
#                            category to the highest confidence
#                            value if above this threshold.
#                            Ex: All biophony categories are 
#                            around 0.2 confidence, but BBI is 0.6.
#                            0.6 > 0.5 threshold, so attach BBI label.
#                            If no categories are 0.5, label "Nothing"
CLASSIFICATION_THRESHOLD = 0.5

# Path to each CNN model (Example format: 'model\\anthro\\model.h5)
ANTHRO_CNN_MODEL = 'model\\anthro\\ant_cnn_model.h5'
BIO_CNN_MODEL = 'model\\bio\\bio_cnn_model.h5'
GEO_CNN_MODEL = 'model\\geo\\geo_cnn_model.h5'

####################################################

################ Removing Warnings #################
#####    Comment out to view any warnings    #######

# Remove tensorflow messages:
# Levels (0,1,2,3): 
#   0     | DEBUG            | [Default] Print all messages       
#   1     | INFO             | Filter out INFO messages           
#   2     | WARNING          | Filter out INFO & WARNING messages 
#   3     | ERROR            | Filter out all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Remove deprecation warnings
warnings.filterwarnings("ignore")

####################################################

class CommandLine:
    def __init__(self):
        # print("Welcome to the Soundscape Noise Analysis Workbench (S.N.A.W.)\n")
        # print("INSTRUCTIONS:\nTo analyze audio, (2) directories are required:\n\t(1) directory for audio files (Ex: 'input')\n\t(1) directory for results of analysis. (Ex: 'output')\n")
        parser = argparse.ArgumentParser(
            description = "Welcome to the Soundscape Noise Analysis Workbench (S.N.A.W.)! S.N.A.W will classify the Biophony, Geophony and Anthrophony in your audio files.")
        parser.add_argument("-i", "--input", help = "Selected file directory for input files (audio file(s) in WAV format)", required = True, default = "")
        parser.add_argument("-o", "--output", help = "Selected file directory for output CSV files", required = True, default = "")

        argument = parser.parse_args()
        status = False
        in_filepath = ""
        out_filepath = ""

        if argument.input:
            print("You have used '-i' or '--input' with argument: {0}".format(argument.input))
            status = True
            in_filepath = argument.input
        if argument.output:
            print("You have used '-o' or '--output' with argument: {0}".format(argument.output))
            status = True
            out_filepath = argument.output
        if not status:
            print("To analyze audio, arguments -i and -o are required. Type -h for help.")

        runStandalone( in_filepath, out_filepath )

def runStandalone(input_filepath, output_filepath):

    # import for loading the cnn models
    from keras.models import load_model

    # load models for classifying all audio files
    if DISPLAY_ALL_STEPS : print("[WORKING] Loading CNN Models..")
    all_models = [ load_model(ANTHRO_CNN_MODEL),
                   load_model(BIO_CNN_MODEL),
                   load_model(GEO_CNN_MODEL) ]
    if DISPLAY_ALL_STEPS : print("[SUCCESS] Loaded CNN Models..")
    
    # Create dictionary for storing return information
    # Initialize a file counter
    finalResult = {}
    fileCount = 0

    # Retrieve File(s) and run classifcation
    for filename in os.listdir(input_filepath):
        audiofile = input_filepath + '/' + filename

        if(os.path.isdir(output_filepath)):
            anthro_csv_file = output_filepath + "/Anthrophony-" + filename[:-4] + ".csv"
            geo_csv_file    = output_filepath + "/Geophony-" + filename[:-4] + ".csv"
            bio_csv_file    = output_filepath + "/Biophony-" + filename[:-4] + ".csv"
            indices_file    = output_filepath + "/Acoustic_Indices-" + filename[:-4] + ".csv"
            csv_file = output_filepath + "/classification_" + filename[:-4] + ".csv"
        else:
            sys.exit("Output filepath not set up correctly. Please retry.")

        csv_columns = ['category','time']
        indice_columns = ['index', 'desc', 'value']
        all_columns = ['category','time', 'index', 'desc', 'value']

        print("\nStarting classification on file: ", filename)

        if DISPLAY_ALL_STEPS : print("[WORKING] Running classification on file ", filename)

        class_data = classify_file( audiofile, all_models )

        anthro_output_dict = class_data[0]["data"]
        geo_output_dict    = class_data[1]["data"]
        bio_output_dict    = class_data[2]["data"]

        if DISPLAY_ALL_STEPS: print("[WORKING] Running indices classification on file ", filename)

        indices_dict = getAcousticIndices(audiofile)
        
        ### Output to CSV ###

        # Output anthrophony csv file
        try:
            with open(anthro_csv_file, 'w') as csvfile_a:
                writer = csv.DictWriter(csvfile_a, fieldnames=csv_columns)
                writer.writeheader()
                for data in anthro_output_dict:
                    writer.writerow(data)
        except IOError:
            print("I/O error in Anthrophony CSV Output")

        # Output geophony csv file
        try:
            with open(geo_csv_file, 'w') as csvfile_g:
                writer = csv.DictWriter(csvfile_g, fieldnames=csv_columns)
                writer.writeheader()
                for data in geo_output_dict:
                    writer.writerow(data)
        except IOError:
            print("I/O error in Geophony CSV Output")

        # Output biophony csv file
        try:
            with open(bio_csv_file, 'w') as csvfile_b:
                writer = csv.DictWriter(csvfile_b, fieldnames=csv_columns)
                writer.writeheader()
                for data in bio_output_dict:
                    writer.writerow(data)
        except IOError:
            print("I/O error in Biophony CSV Output")
        
        if DISPLAY_ALL_STEPS: print("[SUCCESS] Wrote classification results to .csv file")

        # Output acoustic indices csv file
        # try:
        with open(indices_file, 'w') as csvfile_i:
            writer = csv.DictWriter(csvfile_i, fieldnames=indice_columns)
            writer.writeheader()
            for data in indices_dict:
                writer.writerow(data)
        # except IOError:
            # print("I/O error in Acoustic Indice CSV Output")

        if DISPLAY_ALL_STEPS: print("[SUCCESS] Wrote indices classification results to .csv file")
        
        print("Completed classification on file: ", filename)
    
    # Completed classification of all audio files
    print("\n[SUCCESS] All files have been successfuly classified. \nResults are located in: ", output_filepath)

class AudioProcessing(object):

    def __init__(self):
        pass

    @staticmethod
    def convert_to_mono(x):
        if x.ndim > 1:
            return librosa.to_mono(x)
        return x

    @staticmethod
    def get_stft(data,n_fft,win_length,hop_length):
        stft = librosa.stft(y = data,n_fft=n_fft,hop_length=hop_length,win_length=win_length)
        return stft

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b,a = butter(order, normal_cutoff, btype='high', analog=False)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def get_energy(data,frame_length,hop_length):
        energy = librosa.feature.rmse(y=data,n_fft=frame_length,hop_length=hop_length)
        energy = energy[0,:]
        return energy

    @staticmethod
    def resample(data,fs,new_fs):

        # more the sampling rate - more the number of samples in one second
        # Less samples, less quality
        # More samples, good quality, but my lead to more storage requirements.
        # optimal value of sampling rate - 44100 samples per second

        # print("Changing from sampling rate {} to {}".format(fs,new_fs))

        # converting into single channel (monosteric)
        if data.ndim > 1:
            data = librosa.to_mono(data)

        fs = float(fs)
        new_fs = float(new_fs)
        size = data.size

        # old time axis
        old_time_axis = np.arange(size)/fs
        total_time = old_time_axis[-1]
        total_samples = round(total_time*new_fs)

        # getting new time axis wrt old time axis and new sampling rate
        new_time_axis = np.arange(total_samples)/new_fs

        # fills in the values between the samples
        f = interpolate.interp1d(old_time_axis,data)

        new_data = f(new_time_axis)
        return new_data

    @staticmethod
    def rescale(data,scale_range = (-1,1)):
        mini = np.min(data)
        maxi = np.max(data)

        new_min = scale_range[0]
        new_max = scale_range[1]

        new_data = ((new_max - new_min)*(data - mini)/(maxi - mini)) + new_min
        # print("Old min-max :{}-{}, New min-max: {}:{}".format(mini,maxi,new_min,new_max))

        return new_data

    @staticmethod
    def get_histogram(data,bins=np.arange(0,1 + 1.0/256,1.0/256),density = True):

        hist,bins =  np.histogram(data,bins=bins)
        if density == True:
            hist = hist/float(np.sum(hist))

        return hist,bins

    @staticmethod
    def get_entropy(pdf):

        lg = np.log2(pdf)
        lg[np.isneginf(lg)] = 0

        return np.sum(-1.0*pdf*lg)

    @staticmethod
    def get_stft(data,n_fft,win_length,hop_length):
        stft = librosa.stft(y = data,n_fft=n_fft,hop_length=hop_length,win_length=win_length)
        return stft

    @staticmethod
    def get_envelope(data,frame_size = 256):
        total_size = data.size

        envelope = np.copy(data)
        i = 0
        while i < total_size:
            if i + frame_size > total_size:
                envelope[i:] = np.max(abs(data[i:]))
            else:
                envelope[i:i+frame_size] = np.max(data[i:i+frame_size])

            i = i + frame_size

        return envelope


    @staticmethod
    def get_background_noise(envelope):
        envelope = envelope[np.nonzero(envelope)]
        envelope[envelope < 0.001]  = 0

        envelope_non_zero = envelope[np.nonzero(envelope)]
        db_envelope = 20*np.log10(envelope_non_zero)

        sorted_db_envelope = sorted(db_envelope)
        min_index = int(db_envelope.size/200.0)

        db_envelope_min = np.mean(sorted_db_envelope)

        if db_envelope_min < -50:
            db_envelope_min = -50.0

        noise_samples = np.copy(db_envelope)
        # noise_samples[noise_samples > db_envelope_min + 10] = 0

        if ((np.mean(sorted_db_envelope)) - 5 <  (sorted_db_envelope[0] + sorted_db_envelope[-1])/2.0 ) :
            noise_samples[noise_samples > db_envelope_min + 0.4 * ((sorted_db_envelope[-1]) - np.mean(sorted_db_envelope))] = 0
        else:
            noise_samples[noise_samples > db_envelope_min ] = 0


        noise_samples[noise_samples > db_envelope_min + 2*np.std(sorted_db_envelope)] = 0
        noise_samples = noise_samples[np.nonzero(noise_samples)]

        noise_std = np.std(noise_samples)

        try:
            hist,bins = np.histogram(noise_samples,bins=100)
        except:
            import pdb
            pdb.set_trace()

        actual_bins = []
        for i in range(len(bins)-1):
            actual_bins.append((bins[i] + bins[i+1])/2.0)

        new_hist = []
        new_hist.append(hist[0]/3.0)

        for i in range(1,hist.size-1):
            new_hist.append((hist[i-1] + hist[i] + hist[i+1])/3.0)

        new_hist.append(hist[-1]/3.0)
        background_noise = actual_bins[np.argmax(new_hist)] + 0.5*noise_std

        return background_noise

    @staticmethod
    def get_row_background_noise(envelope):
        envelope = envelope[np.nonzero(envelope)]

        envelope_min = np.min(envelope)

        envelope_non_zero = envelope[np.nonzero(envelope)]
        db_envelope = 20 * np.log10(envelope_non_zero)

        db_envelope_min = 20 * np.log10(envelope_min)

        if db_envelope_min < -60:
            db_envelope_min = -60.0

        noise_samples = np.copy(db_envelope)
        noise_samples[noise_samples > db_envelope_min + 10] = 0
        noise_samples = noise_samples[np.nonzero(noise_samples)]

        noise_std = np.std(noise_samples)

        try:
            hist, bins = np.histogram(noise_samples, bins=100)
        except:
            import pdb
            pdb.set_trace()

        actual_bins = []
        for i in range(len(bins) - 1):
            actual_bins.append((bins[i] + bins[i + 1]) / 2.0)

        new_hist = []
        new_hist.append(hist[0] / 3.0)

        for i in range(1, hist.size - 1):
            new_hist.append((hist[i - 1] + hist[i] + hist[i + 1]) / 3.0)

        new_hist.append(hist[-1] / 3.0)

        background_noise = actual_bins[np.argmax(new_hist)] + 0.2 * noise_std

        return background_noise

# CLassify Files with CNNs
def classify_file(audio_file, all_models) :

    all_labels = [ ["AAT", "AHV", "AMA", "ART", "ASI", "AVH", "AVT"],
                   ["BRA", "BAM", "BBI", "BMA", "BIN"],
                   ["GOC", "GRA", "GST","GWG", "GWC"] ]

    classify_dict = [ {'name' : 'Anthrophony',
                       'color' : '#0088FE',
                       'data' : [] },
                      {'name': 'Biophony',
                       'color': '#00C49F',
                       'data': [] },
                      {'name': 'Geophony',
                       'color': '#FFBB28',
                       'data': [] } ]

    n_mfcc = 128 # bucket size
    max_len = 32 # max_len size
    channels = 1 # channels

    # convert file to wav2mfcc
    # Mel-frequency cepstral coefficients
    file_path = audio_file
    big_wave, sr = librosa.load(file_path, mono=True, sr=None)

    for sec_index in range( int(big_wave.shape[0] / sr) ) :
        start_sec = sec_index
        end_sec = sec_index + 1

        sec_to_trim = np.array( [ float(start_sec), float(end_sec) ] )
        sec_to_trim = np.ceil( sec_to_trim * sr )

        wave = big_wave[int(sec_to_trim[0]) : int(sec_to_trim[1])]

        wave = np.asfortranarray(wave[::3])
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)

        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if (max_len > mfcc.shape[1]):
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :max_len]

        # Convert wav to MFCC
        prediction_data = mfcc

        # Reshape to 4 dimensions
        prediction_data = prediction_data.reshape(1, n_mfcc, max_len, channels)

        # Run the model on the inputted file

        all_predicted = [ model.predict(prediction_data) for model in all_models ]

        for labels, predicted, classification in zip( all_labels, all_predicted, classify_dict ) :
            if( PREDICTION_VERBOSE ):
                print ('\nPREDICTED VALUES: ')
            labels_indices = range(len(labels))
            max_value = 0
            max_value_index = 0
            for index in labels_indices:
                # print the predicted values for each category at each timestamp
                if( PREDICTION_VERBOSE ):
                    print("\n", labels[index], ": ", '%.08f' % predicted[0,index])
                if predicted[0,index] > max_value:
                    max_value_index = index
                    max_value = predicted[0,index]

            # Output the prediction
            if max_value < CLASSIFICATION_THRESHOLD:
                if( PREDICTION_VERBOSE ):
                    print("\nGUESS: Nothing")
                classification['data'].append( { "category" : "NO", "time" : start_sec } )
            else:
                if( PREDICTION_VERBOSE ):
                    print('\nGUESS: ', labels[max_value_index])
                classification['data'].append( { "category" : labels[max_value_index], "time" : start_sec } )
    if( PREDICTION_VERBOSE ):
        print(classify_dict)

    return classify_dict


'''
###------------------------------------------------------###
Function: AcousticIndices(object)
###------------------------------------------------------###
CREDIT:
The below code is from the user "amogh3892" with repo
"Acoustic-Indices." This set of files is being treated
as a library to be used by our product.
###------------------------------------------------------###

'''
class AcousticIndices(object):

    def __init__(self,data, fs,n_fft = 512, win_len = 512, hop_len = 512):

        chop_fraction = data.size % win_len

        if chop_fraction != 0 :
            data = data[:-chop_fraction]

        self.data = data
        self.fs = fs

        self.envelope = AudioProcessing.get_envelope(self.data,frame_size=win_len)
        self.envelope[self.envelope < 0 ] = 0

        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len

        self.stft = AudioProcessing.get_stft(self.data,n_fft,win_len,hop_len)
        self.stft = np.absolute(self.stft)
        self.smooth_spectrogram()

        n_drop = int(np.ceil(500/(float(self.fs)/self.n_fft)))
        self.stft = self.stft[n_drop:,:]

        self._remove_spectrogram_noise()


        self._calculate_background_noise()
        self._get_temporal_entropy()

        self._get_segments_above_noise()
        self._get_spectral_entropy()


    def smooth_spectrogram(self,average_filter_radius = (1,1)):
        stft_img = sitk.GetImageFromArray(self.stft)
        avg_stft = sitk.Mean(stft_img,average_filter_radius)
        self.stft = sitk.GetArrayFromImage(avg_stft)

    def _remove_spectrogram_noise(self):

        new_spec = np.zeros(self.stft.shape)

        for i in range(self.stft.shape[0]):
            row = self.stft[i,:]
            bn_log = AudioProcessing.get_row_background_noise(row)
            bn = 10**(bn_log/20.0)
            row = row - bn
            row[row < 0] = 0
            new_spec[i,:] = row

        self.stft = new_spec


    @staticmethod
    def get_normalized_value(value,norm = (0,1)):

        if value < norm[0]:
            return 0
        elif value > norm[1]:
            return 1
        else:
            return ((value - float(norm[0]))/(norm[1] - float(norm[0])))

    def _get_segments_above_noise(self):
        threshold = self.background_noise + 3
        envelope = AudioProcessing.get_envelope(abs(self.data), frame_size=1024)
        non_zero = envelope[np.nonzero(envelope)]
        data_log = 20*np.log10(non_zero)

        ind = np.where(data_log > threshold)

        check_array = np.zeros(self.envelope.size)
        check_array[ind] = 1


        diff = np.diff(check_array)

        ones = np.where(diff == 1)[0]
        minus_ones = np.where(diff == -1)[0]

        if ones.size == 0:
            ones = np.array([0])

        if minus_ones.size == 0:
            minus_ones = np.array([check_array.size - 1])

        if ones[0] >= minus_ones[0]:
            ones = np.append(0,ones)

        if ones[-1] >= minus_ones[-1]:
            minus_ones = np.append(minus_ones,[check_array.size - 1])

        segments = []

        # considering segments which are greater than 100 ms
        min_seg_length = 0.1*self.fs

        for i in range(ones.size):
            seg_duration = (minus_ones[i]-ones[i])

            if seg_duration > min_seg_length:
                segments.append((ones[i],minus_ones[i],minus_ones[i] - ones[i]))

        if 0:
            # plotting check array

            segment_array = np.zeros(self.data.size)
            for segment in segments:
                segment_array[segment[0]:segment[1]] = 1
            import matplotlib.pyplot as plt
            plt.plot(self.data)
            plt.plot(segment_array)
            plt.show()


        self.segments = segments


    def _get_temporal_entropy(self):
        envelope = self.envelope[np.nonzero(self.envelope)]
        envelope_energy = envelope**2

        N = 2**10 # (i.e 1024 values of envelope energy possible)

        envelope_energy = AudioProcessing.rescale(envelope_energy,(0,N))
        envelope_energy = envelope_energy.astype(np.uint16)

        item_frequency = itemfreq(envelope_energy)

        total_samples = envelope_energy.size

        pmf = []

        for i in range(item_frequency.shape[0]):
            pmf.append(item_frequency[i][1])

        pmf = np.array(pmf)
        pmf = pmf/float(total_samples)

        self.temporal_entropy = AudioProcessing.get_entropy(pmf)/np.log2(N)


    def get_temporal_entropy(self):
        return self.temporal_entropy

    def _get_spectral_entropy(self):

        stft = np.copy(self.stft)
        N = 2**10
        stft = AudioProcessing.rescale(stft,(0,N))
        stft = stft.astype(np.uint16)

        item_frequency = itemfreq(stft)

        total_samples = stft.size

        pmf = []

        for i in range(item_frequency.shape[0]):
            pmf.append(item_frequency[i][1])

        pmf = np.array(pmf)
        pmf = pmf/float(total_samples)

        self.spectral_entropy = AudioProcessing.get_entropy(pmf)/np.log2(N)

    def get_spectral_entropy(self):
        return self.spectral_entropy

    def get_acoustic_entropy(self):
        return self.temporal_entropy * self.spectral_entropy

    def get_average_signal_amplitude(self,norm=(-50,-3)):
        avg_envelope = np.mean(self.envelope)
        db_avg_envelope = 20*np.log10(avg_envelope)
        db_avg_envelope_normalized = AcousticIndices.get_normalized_value(db_avg_envelope,norm=norm)
        return db_avg_envelope_normalized

    def _calculate_background_noise(self):

        background_noise = AudioProcessing.get_background_noise(self.envelope)
        self.background_noise = background_noise

    def get_background_noise(self,norm = (-50,-3)):

        background_noise = self.background_noise

        background_noise_normalized = AcousticIndices.get_normalized_value(background_noise,norm = norm)
        return background_noise_normalized

    def get_snr(self,norm = (3,50)):
        max_envelope = np.max(self.envelope)
        db_max_envelope = 20*np.log10(max_envelope)
        snr = db_max_envelope - self.background_noise
        snr_normalized = AcousticIndices.get_normalized_value(snr,norm = norm)
        return snr_normalized

    def get_acoustic_activity(self,norm =(0,1)):

        total_samples = self.envelope.size
        samples_above_threshold = 0
        for segment in self.segments:
            samples_above_threshold = samples_above_threshold + segment[2]

        acoustic_activity = samples_above_threshold/float(total_samples)
        acoustic_activity_normalized = AcousticIndices.get_normalized_value(acoustic_activity,norm=norm)
        return acoustic_activity_normalized

    def get_acoustic_activity_count(self,norm = (0,140)):
        acoustic_activity_count = len(self.segments)
        acoustic_activity_count = AcousticIndices.get_normalized_value(acoustic_activity_count,norm=norm)
        return acoustic_activity_count

    def get_acoustic_events_average_duration(self):

        segments_duration = []

        for segment in self.segments:
            segments_duration.append(segment[2])

        average_segments_duration = np.mean(segments_duration)
        average_segments_duration_ms = (average_segments_duration/float(self.fs))*1000.0

        average_segments_duration_ms = AcousticIndices.get_normalized_value(average_segments_duration_ms,(0,3000))

        return average_segments_duration_ms

    def get_mid_band_activity(self):
        stft = np.copy(self.stft)

        frequency_resolution = self.fs/self.n_fft

        no_frequency_bins = int(3000/frequency_resolution)

        stft = stft[:no_frequency_bins,:]

        spec_1d = stft.ravel()
        threshold = 0.015 # experimentally calculated
        total_spectrogram_pixels = spec_1d.size
        ind_above_threshold = np.where(spec_1d > threshold)
        total_pixels_above_threshold = ind_above_threshold[0].size
        mid_band_activity = total_pixels_above_threshold/float(total_spectrogram_pixels)
        return mid_band_activity

    def get_spectral_maxima_entropy(self):
        max_bins = []

        for segment in self.segments:
            start = int(float(segment[0])/self.n_fft)
            stop = int(float(segment[1])/self.n_fft)

            for i in range(start,stop+1):
                stft_column = self.stft[:,i]
                max_bins.append(np.argmax(stft_column))


        pdf,bins = AudioProcessing.get_histogram(max_bins,bins = np.arange(0,self.stft.shape[0]))

        pdf = pdf[np.nonzero(pdf)]
        spectral_max_entropy = AudioProcessing.get_entropy(pdf)/np.log2(self.stft.shape[0])

        return spectral_max_entropy

    def get_spectral_average_variance_entropy(self):

        segments_stft = None

        N = 2**10

        stft = np.copy(self.stft)

        for segment in self.segments:
            start = int(float(segment[0])/self.n_fft)
            stop = int(float(segment[1])/self.n_fft)

            if segments_stft is None:
                segments_stft = stft[:,start:stop]
            else:
                segments_stft = np.concatenate((segments_stft,stft[:,start:stop]),axis=1)


        average_spectra = np.mean(segments_stft,axis=1)
        var_spectra = np.var(segments_stft,axis =1)

        N = 2**8
        average_spectra = AudioProcessing.rescale(average_spectra,(0,N))
        average_spectra = average_spectra.astype(np.uint8)

        var_spectra = AudioProcessing.rescale(var_spectra,(0,N))
        var_spectra = var_spectra.astype(np.uint8)


        avg_pdf,bins = AudioProcessing.get_histogram(average_spectra,bins=np.arange(0,255))
        var_pdf,bins = AudioProcessing.get_histogram(var_spectra,bins=np.arange(0,255))


        avg_pdf = avg_pdf[np.nonzero(avg_pdf)]
        var_pdf = var_pdf[np.nonzero(var_pdf)]

        average_spectrum_entropy = AudioProcessing.get_entropy(avg_pdf)/np.log2(N)
        variance_spectrum_entropy = AudioProcessing.get_entropy(var_pdf)/np.log2(N)

        return average_spectrum_entropy,variance_spectrum_entropy

    def get_soundscape_indices(self,total_bins = 7):

        biophony_levels = []

        stft = np.copy(self.stft)
        n_drop = int(np.ceil(self.fs/self.n_fft))
        stft = stft[n_drop:,:]

        frequency_interval = int(np.floor(stft.shape[0]/float(total_bins)))
        extra = stft.shape[0] % frequency_interval

        max_bin = stft.shape[0] - extra

        psd = (1.0 / self.win_len)*(self.stft**2)
        # psd = AudioProcessing.rescale(psd,(0,1))
        psd_sum = np.sum(psd)

        for i in range(0,max_bin,frequency_interval):
            bin = psd[i:i + frequency_interval, :]
            biophony_levels.append(np.sum(bin) / psd_sum)

        a = biophony_levels[0]
        b = np.max(biophony_levels[1:])

        biophony = b
        anthrophony = a
        # rho = b/float(a)
        ndsi = (b-a)/(b+a)
        ndsi = AcousticIndices.get_normalized_value(ndsi,(-1,1))

        return anthrophony,biophony,ndsi


    def get_acoustic_complexity_index(self,time_interval_sec = 5.0):
        stft = self.stft

        samples_in_interval = int(self.fs/float(self.win_len)*time_interval_sec)
        total_samples = stft.shape[1]
        step1_matix = None

        i = 0
        while i < total_samples:
            if i + samples_in_interval > total_samples:
                # sub_diff = np.abs(diff_stft[:,i:])
                sub_stft = stft[:,i:]
                sub_diff = np.abs(np.diff(sub_stft))

            else:
                # sub_diff = np.abs(diff_stft[:,i:i+samples_in_interval])
                sub_stft = stft[:, i:i + samples_in_interval]
                sub_diff = np.abs(np.diff(sub_stft))


            step1_diff = np.sum(sub_diff,axis = 1)
            step1_stft = np.sum(sub_stft,axis = 1)

            step1_stft[step1_stft == 0] = 1

            step1 = step1_diff/step1_stft

            step1 = np.expand_dims(step1,axis=1)

            if step1_matix is None:
                step1_matix = step1
            else:
                step1_matix = np.concatenate((step1_matix,step1),axis=1)


            i = i + samples_in_interval

        aci = np.sum(step1_matix)
        aci = AcousticIndices.get_normalized_value(aci,(0.0,step1_matix.size))

        return aci




    def get_shannon_index(self,total_bins = 8):

        frequency_interval = int(np.ceil(1000.0/(self.fs/float(self.n_fft))))
        bin_probs = []

        i = 0

        psd = (1.0 / self.win_len)*(self.stft**2)
        psd = AudioProcessing.rescale(psd,(0,1))

        psd_sum = np.sum(psd)


        while i < total_bins*frequency_interval :

            if i + frequency_interval > total_bins*frequency_interval:
                bin = psd[i:, :]
            else:

                bin = psd[i:i + frequency_interval, :]

            bin_probs.append(np.sum(bin))

            i = i + frequency_interval


        bin_probs = np.array(bin_probs)
        bin_probs = bin_probs/psd_sum

        shannon_index = AudioProcessing.get_entropy(bin_probs)/np.log2(len(bin_probs))
        return shannon_index


    def get_median_amplitude_envelope(self,norm=(-50,-3)):
        envelope = self.envelope
        envelope = envelope[np.nonzero(envelope)]
        median_envelope = np.median(envelope)
        db_median_envelope = 20 * np.log10(median_envelope)
        db_median_envelope_normalized = AcousticIndices.get_normalized_value(db_median_envelope, norm=norm)
        return db_median_envelope_normalized

    def get_number_of_peaks(self):
        stft = np.copy(self.stft)
        stft = AudioProcessing.rescale(stft,(0,1))

        no_peaks = 0

        min_distance = np.ceil(int(np.ceil(200.0/(self.fs/float(self.n_fft))))/3.0)

        for i in range(stft.shape[1]):
            col = stft[:,i]

            import pdb
            pdb.set_trace()

            peaks = peakutils.indexes(col,thres=0.3,min_dist= min_distance)

            for peak in peaks:
                if peak != 0:
                    if col[peak] - col[peak -1]> 0.01:
                        no_peaks = no_peaks + 1

        return no_peaks


    def get_spectral_diversity_persistance(self):
        stft = np.copy(self.stft)
        extra = stft.shape[0] % 3

        if extra !=0 :
            stft = stft[:-extra,:]

        new_stft = None
        for i in range(0,stft.shape[0],3):
            bin = stft[i:i+3,:]
            bin = np.mean(bin,axis=0)
            bin = np.column_stack((bin))
            if new_stft is None:
                new_stft = bin
            else:
                new_stft = np.concatenate((new_stft,bin))

        new_stft[new_stft <= 0.07] = 0
        new_stft[new_stft > 0.07] = 1

        training_data = None

        for i in range(new_stft.shape[1]):
            spectra = new_stft[:,i]
            for j in range(1,spectra.size -1):
                if spectra[j] == 1 and (spectra[j-1] == 0 and spectra[j+1] == 0):
                    spectra[j] = 0

            spectra = np.column_stack((spectra))

            if np.where(spectra == 1)[0].size >= 2:
                if training_data is None:
                    training_data = spectra
                else:
                    training_data = np.concatenate((training_data,spectra))

        cluster_count = 1
        similarity_threshold = 0.25
        if training_data.shape[0] < 9 :
            return (0,0)
        else:

            clusters = training_data[:2,:]

            training_data = training_data[1:,:]
            sim_scores = []
            for i in range(training_data.shape[0]):
                sim_scores.append(np.sum(np.logical_and(training_data[i,:],clusters[0,:]))/float(np.sum(np.logical_or(training_data[i,:],clusters[0,:]))))

            min_sim = np.argmin(sim_scores)
            clusters[1,:] = training_data[min_sim,:]

            training_data = np.delete(training_data,min_sim,0)

            clusters = training_data[:2,:]
            training_data = training_data[2:,:]
            quantized_value = [0,1]
            for i in range(training_data.shape[0]):
                similarity_scores = []
                for j in range(clusters.shape[0]):

                    similarity_score = np.sum(np.logical_and(training_data[i,:],clusters[j,:]))/float(np.sum(np.logical_or(training_data[i,:],clusters[j,:])))
                    similarity_scores.append(similarity_score)


                max_similarity_score = np.max(similarity_scores)
                max_similarity_index = np.argmax(similarity_scores)
                if max_similarity_score > similarity_threshold:
                    quantized_value.append(max_similarity_index)
                else:
                    new_cluster = np.column_stack(training_data[i,:])
                    clusters = np.concatenate((clusters,new_cluster))
                    cluster_count = cluster_count + 1
                    quantized_value.append(cluster_count)



        run_lengths = []
        for i in range(len(quantized_value)):
            if i ==0:
                count = 1
                previous_value = quantized_value[i]
            else:
                value = quantized_value[i]
                if value == previous_value:
                    count = count + 1
                    previous_value = value
                else:
                    run_lengths.append(count)
                    count = 1
                    previous_value = value


        # removing elements occuring with run_length = 1
        newRunLengths = []
        for i in run_lengths:
            if i == 1:
                continue
            else:
                newRunLengths.append(i)

        spectral_diversity = cluster_count
        spectral_persistance = np.mean(newRunLengths)

        spectral_diversity = AcousticIndices.get_normalized_value(spectral_diversity,(0,50))
        spectral_persistance = AcousticIndices.get_normalized_value(spectral_persistance,(0,20))

        return spectral_diversity,spectral_persistance



    def get_acoustic_indices(self):
        feature_vector = []
        feature_vector.append(self.get_snr())
        feature_vector.append(self.get_acoustic_activity())
        feature_vector.append(self.get_acoustic_activity_count())
        feature_vector.append(self.get_acoustic_events_average_duration())
        feature_vector.append(self.get_temporal_entropy())
        feature_vector.append(self.get_spectral_entropy())
        feature_vector.append(self.get_acoustic_entropy())
        feature_vector.extend(self.get_soundscape_indices())
        feature_vector.append(self.get_acoustic_complexity_index())
        feature_vector.append(self.get_shannon_index())
        feature_vector.append(self.get_median_amplitude_envelope())
        feature_vector.append(self.get_spectral_maxima_entropy())
        feature_vector.extend(self.get_spectral_average_variance_entropy())

        return feature_vector

    @staticmethod
    def get_acoustic_indices_headers():
        feature_headers = []
        feature_headers.append("SNR")
        feature_headers.append("Acoustic Activity")
        feature_headers.append("Acoustic Activity Count")
        feature_headers.append("Acoustic Events Average Duration")
        feature_headers.append("Temporal Entropy")
        feature_headers.append("Spectral Entropy")
        feature_headers.append("Acoustic Entropy")
        feature_headers.extend(["Anthrophony","Biophony","Normalized Difference Soundscape Index"])
        feature_headers.append("Acoustic Complexity Index")
        feature_headers.append("Shannon Index")
        feature_headers.append("Median Of Amplitude Envelope")
        feature_headers.append("Entropy Of Spectral Maxima")
        feature_headers.append("Entropy Of Spectral Average")
        feature_headers.append("Entropy Of Spectral Variance")

        return feature_headers

    @staticmethod
    def get_acoustic_indices_descs():
        feature_descs = []
        feature_descs.append("The decibel difference of the highest amplitude acoustic event and the background noise.")
        feature_descs.append("Fraction of frames where the signal is >3dB above background. ")
        feature_descs.append("The number of times acoustic activity events occurred. ")
        feature_descs.append("The mean length of acoustic activity events. ")
        feature_descs.append("[Ht] Complex amplitude signal - tends towards 1 and is sensitive to consistent sound. ")
        feature_descs.append("[Hs] or [Hf] complex frequency signal - tends towards 1 and is sensitive to consistent sound. ")
        feature_descs.append("[H] assessment of pure to random noise: [Hs] x [Ht]. ")
        feature_descs.extend(["Anthrophony Desc","Biophony Desc","ratio of human to biotic sound - values of 1 equate to no human sound. "])
        feature_descs.append("Heterogeneity in sound intensity - high values represent more dynamically changing sounds. ")
        feature_descs.append("Entropy of frequency - noisy will tend towards 1. ")
        feature_descs.append("Linearly related to number of animal vocalizations, part of acoustic richness. ")
        feature_descs.append("Indicator of species richness. ")
        feature_descs.append("Entropy Of Spectral Average Desc. ")
        feature_descs.append("Indicator of species richness. ")

        return feature_descs

'''
###------------------------------------------------------###
Function: getAcousticIndices(fileOrDirectory, isDirectory)
Params: 
- fileOrDirectory = file path OR directory with .WAV files
- isDirectroy = OPTIONAL set to False. This must be called True 
                if passing a directory instead of a single file.
Caller: Api.py (Not yet)
###------------------------------------------------------###
This function begins the process of generating the Acoustic
Indices on a specific file or directory of files.
###------------------------------------------------------###
CREDIT:
The above code is from the user "amogh3892" with repo
"Acoustic-Indices." This set of files is being treated
as a library to be used by our product.
###------------------------------------------------------###

'''
def getAcousticIndices(audiofile):
    if( DISPLAY_ALL_STEPS ): print("[WORKING] Attempting to run acoustic indices calculator..")

    # loop through the files in the directory
    try:
        data,fs  =  librosa.load(audiofile, sr=None, offset=0, duration=60)
        # mono channel
        data = AudioProcessing.convert_to_mono(data)

        # changing sampling rate
        new_fs = 17640
        data_chunk = AudioProcessing.resample(data,fs,new_fs)

        # extracting indices
        acousticIndices = AcousticIndices(data_chunk,new_fs)
        acoustic_indices = acousticIndices.get_acoustic_indices()

        acoustic_indices = list(map(lambda x: round(x, 4), acoustic_indices))
        if(PREDICTION_VERBOSE):
            print("\nAcoustic Indice Values:\n", acoustic_indices, "\n")

        acoustic_headers = acousticIndices.get_acoustic_indices_headers()
        acoustic_descs = acousticIndices.get_acoustic_indices_descs()
        # singleResultArray is used to store the results of one file (List of dictionaries)
        singleResultArray = []

        # Traverse the acoustic tags
        for i in range(len(acoustic_headers)):
            # per indices in the length of the acoustic tags,
            # append dictionary items.
            singleResultArray.append({"index": acoustic_headers[i], "value" : acoustic_indices[i], "desc" : acoustic_descs[i]})

            # append result dictionary to the final results array
            if( DISPLAY_ALL_STEPS ): print("[WORKING] Calculated " + acoustic_headers[i])
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        singleResultArray = "ERROR_PRESENT"


    if( DISPLAY_ALL_STEPS ):
        print("[SUCCESS] Calculated all acoustic indices")
    return singleResultArray

if __name__ == '__main__':
   app = CommandLine()
