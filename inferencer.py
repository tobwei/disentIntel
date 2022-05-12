import torch
from scipy import signal
import numpy as np
from numpy.random import RandomState
import soundfile as sf
from pysptk import sptk
from librosa.filters import mel
from hparams import hparams
from utils import butter_highpass, pad_seq_to_2, pySTFT, quantize_f0_numpy, speaker_normalization
from model import Generator_3 as Generator
from utils import pad_f0



class Inferencer(object):
	def __init__(self, model_path):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.hparams = hparams
		self.G_path = model_path
		self.G = self.load_models()
		self.mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
		self.min_level = np.exp(-100 / 20 * np.log(10))
		self.b, self.a = butter_highpass(30, 1600, order=5)


	def load_models(self):
		G = Generator(self.hparams).eval().to(self.device)
		g_checkpoint = torch.load(self.G_path, map_location=lambda storage, loc: storage)
		G.load_state_dict(g_checkpoint['model'])
		return G


	def load_sample(self, sample_path, speaker_id, gender=None):
		self.speaker_id = speaker_id
		self.prng = RandomState(int(self.speaker_id))
		_, _, mspec, f0_norm = self.gen_mel_f0(sample_path, gender)
		return mspec, f0_norm


	def prepare_data(self, mspec, f0_norm):
		max_len = self.hparams.max_len_pad
		mspec_pad = self.pad_utt(mspec, max_len)

		f0_norm_pad = pad_f0(f0_norm.squeeze(), max_len)
		f0_norm_pad_quant = quantize_f0_numpy(f0_norm_pad)[0]
		f0_norm_pad_onehot = f0_norm_pad_quant[np.newaxis,:,:]
		f0_norm_pad_onehot = torch.from_numpy(f0_norm_pad_onehot).to(self.device)

		return mspec_pad, f0_norm_pad_onehot


	def inference(self, mspec_pad, f0_norm_pad_onehot, speaker_dim):
		uttr = mspec_pad.type(torch.float32).to(self.device)
		uttr_f0 = torch.cat((uttr, f0_norm_pad_onehot), dim=-1)
		emb_empty = torch.from_numpy(np.zeros((1,speaker_dim), dtype=np.float32)).to(self.device)
		mel_out = self.G(uttr_f0, uttr, emb_empty)
		return mel_out


	def gen_mel_f0(self, path, gender):
		if gender == 'M':
			lo, hi = 50, 250
		elif gender == 'F':
			lo, hi = 100, 600
		# read audio file
		x, sr = sf.read(path)
		assert sr == 16000, 'Sample rate has to be 16kHz.'
		if x.shape[0] % 256 == 0:
			x = np.concatenate((x, np.array([1e-06])), axis=0)
		y = signal.filtfilt(self.b, self.a, x)
		wav = y * 0.96 + (self.prng.rand(y.shape[0])-0.5)*1e-06
		# generate spectrogram
		D = pySTFT(wav).T
		D_mel = np.dot(D, self.mel_basis)
		D_db = 20 * np.log10(np.maximum(self.min_level, D_mel)) - 16
		S = (D_db + 100) / 100
		# extract f0 (pitch contour)
		f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, sr, 256, min=lo, max=hi, otype=2)
		index_nonzero = (f0_rapt != -1e10)
		mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
		f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

		assert len(S) == len(f0_rapt), 'length of melspec and pitch contour do not match'
		return wav, sr, S, f0_norm


	def pad_utt(self, utterance, len_out=1024):
		utt_pad, _ = pad_seq_to_2(utterance[np.newaxis,:,:], len_out)
		utt_pad = torch.from_numpy(utt_pad).to(self.device)
		return utt_pad


