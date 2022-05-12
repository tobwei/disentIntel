from json.tool import main
import statistics
import random
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import scipy
from logging import root
import librosa
import librosa.display
import os
from shutil import copyfile
from librosa.sequence import dtw
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
import csv
import soundfile
from inferencer import Inferencer
from assets.py_vad_tool.audio_tools import * 
from sox import file_info
from assets.py_vad_tool.unsupervised_vad import compute_log_nrg, nrg_vad



def create_UASpeech_custom_v2():
        """
        1) Creates a 'usable' UASpeech dataset, used for the intelligibility evaluation.
        """
        
        # creating a list with all 765 words per speaker
        word_file = '/Users/tobi/Data/corpora/UASPEECH_SAGI/transcription_lut.csv'
        wordlist = []
        with open(word_file, newline='') as handle:
                reader = csv.reader(handle)
                for row in reader:
                        wordlist.append(row[1])
        wordlist = wordlist[1:]
        full_wordlist = []
        for word in wordlist:
                if '_' in word:
                        full_wordlist.append(word)
                else:
                        full_wordlist.append('B1_' + word)
                        full_wordlist.append('B2_' + word)
                        full_wordlist.append('B3_' + word)

        # filling a dictionary where speaker is key, with one valid file for each of the 765 utterances, where the mic is chosen randomly
        source_dir = '/Users/tobi/Data/corpora/UASPEECH_SAGI/audio'
        results = {}
        _, dirs, _ = next(os.walk(source_dir))
        for dir in sorted(dirs):
                zero_count = 0
                speaker = dir
                print(f'Finding valid files for speaker {speaker} ... ')
                valid_files = []
                _, _, files = next(os.walk(os.path.join(source_dir, dir)))
                for word in full_wordlist:
                        for file in sorted(files):
                                # try one randomly selected channel, ...
                                mic = np.random.randint(2,9)
                                target = speaker + '_' + word + '_M' + str(mic) + '.wav'
                                target_path = os.path.join(source_dir, dir, target)
                                if not os.path.exists(target_path):
                                        # can happen that the initial mic doesnt exist
                                        while not os.path.exists(target_path):
                                                mic = np.random.randint(2,9)
                                                target = speaker + '_' + word + '_M' + str(mic) + '.wav'
                                                target_path = os.path.join(source_dir, dir, target)
                                if os.path.getsize(target_path) == 0:
                                        # if that didn't work, try until you find one that works
                                        while os.path.getsize(target_path) == 0:
                                                mic = np.random.randint(2,9)
                                                target = speaker + '_' + word + '_M' + str(mic) + '.wav'
                                                target_path = os.path.join(source_dir, dir, target)
                                                if not os.path.exists(target_path):
                                                        while not os.path.exists(target_path):
                                                                mic = np.random.randint(2,9)
                                                                target = speaker + '_' + word + '_M' + str(mic) + '.wav'
                                                                target_path = os.path.join(source_dir, dir, target)
                                # print(f'Found a valid file: {target_path}')
                                valid_files.append(target_path)
                                break
                results[speaker] = valid_files

        # delte 3 files from every speaker (765 - 3 = 762) because for patient M01, those files are 30+ seconds long and contain only background noise pretty much
        # delte 1 file from every speaker as well, from patient M10, since just blabbering with somebody
        for speaker in results:
                for file in results[speaker]:
                        if 'B1_CW53' in file:
                                results[speaker].remove(file)
                        if 'B2_D3' in file:
                                results[speaker].remove(file)
                        if 'B3_UW84' in file:
                                results[speaker].remove(file)
                        if 'B1_UW3_' in file:
                                results[speaker].remove(file)
                        if 'B1_UW53' in file:
                                results[speaker].remove(file)

        # copying all the files in the dict to the new target location
        target_dir = '/Users/tobi/Data/corpora/UASPEECH_Custom_3/audio'
        for speaker in results:
                print(f'Copying valid files for speaker {speaker} ... ')
                if not os.path.exists(os.path.join(target_dir, speaker)):
                        os.makedirs(os.path.join(target_dir, speaker))
                for file in results[speaker]:
                        file_name = file.split('/')[-1]
                        src = file
                        dest = os.path.join(target_dir, speaker, file_name) 
                        copyfile(src, dest)



def cut_vad(c, nrg):
        """
        2) Perform 'cut + vad' for all files in UASpeech_Custom
        """
        root_dir = '/Users/tobi/Data/corpora/UASPEECH_Custom_3/audio'
        _, dirs, _ = next(os.walk(root_dir))
        for dir in sorted(dirs):
                print(f'Cutting and VADing files from speaker {dir} at the moment ... ')
                _, _, files = next(os.walk(os.path.join(root_dir, dir)))
                for file in sorted(files):
                        if '.wav' in file:
                                file_path = os.path.join(root_dir, dir, file)
                                fs, s = read_wav(file_path)
                                # cut c at start and end
                                duration = len(s) / fs 
                                cut_p = duration * c
                                cut_p_frames = int(cut_p * fs)
                                cut_start = cut_p_frames
                                cut_end = len(s) - cut_p_frames
                                s = s[cut_start:cut_end]
                                # vad
                                win_len = int(fs*0.025)
                                hop_len = int(fs*0.010)
                                sframes = enframe(s, win_len, hop_len)
                                vad = nrg_vad(sframes, nrg, context=5)  #v2:context=20
                                energy_frames = deframe(vad, win_len, hop_len)
                                # create new audio file, with only selected energy frames
                                new_frames = []
                                for i in range(len(s)):
                                        if energy_frames[i] == 1.:
                                                new_frames.append(s[i])
                                os.remove(file_path)
                                soundfile.write(file_path, new_frames, samplerate=16000)



def cut_vad_single(c, nrg, file_path):
        """
        Cut and VAD a single file, given by path.
        """
        
        fs, s = read_wav(file_path)
        # cut c at start and end
        duration = len(s) / fs 
        cut_p = duration * c
        cut_p_frames = int(cut_p * fs)
        cut_start = cut_p_frames
        cut_end = len(s) - cut_p_frames
        s = s[cut_start:cut_end]
        # vad
        win_len = int(fs*0.025)
        hop_len = int(fs*0.010)
        sframes = enframe(s, win_len, hop_len)
        vad = nrg_vad(sframes, nrg, context=5)  #v2:context=20
        energy_frames = deframe(vad, win_len, hop_len)
        # create new audio file, with only selected energy frames
        new_frames = []
        for i in range(len(s)):
                if energy_frames[i] == 1.:
                        new_frames.append(s[i])
        os.remove(file_path)
        soundfile.write(file_path, new_frames, samplerate=16000)



def inference():
        """
        3) Inference with SpeechSplit to save 3 codes per utterance
        """

        weights = 'assets/weights/en/805000-G.ckpt'
        spk_emb_dim = 14212
        infer = Inferencer(weights)
        # audio_dir = '/Users/tobi/Data/corpora/UASPEECH_Custom_2/audio'
        # meta_dir = '/Users/tobi/Data/corpora/UASPEECH_Custom_2/inference'
        audio_dir = '/DATA/tobi/datasets/UASPEECH_Custom_3/audio'
        meta_dir = '/DATA/tobi/datasets/UASPEECH_Custom_3/inference'
        _, dirs, _ = next(os.walk(audio_dir))

        for dir in sorted(dirs):
                gender = 'M' if 'M' in dir else 'F'
                _, _, files = next(os.walk(os.path.join(audio_dir, dir)))
                for file in sorted(files):
                        if '.wav' in file:
                                file_path = os.path.join(audio_dir, dir, file)
                                # skip folders that already exist
                                file = file_path.split('/')[-1][:-4]
                                meta_path = os.path.join(meta_dir, dir, file)
                                # if inference files do not already exist, create them
                                if not os.path.exists(meta_path):
                                        # inferece
                                        print(file)
                                        mspec, f0_norm = infer.load_sample(file_path, spk_emb_dim, gender)
                                        mspec_pad, f0_norm_pad_onehot = infer.prepare_data(mspec, f0_norm)
                                        inf_out = infer.inference(mspec_pad, f0_norm_pad_onehot, spk_emb_dim)
                                        
                                        #codes
                                        mel_length = mspec.shape[0]
                                        with torch.no_grad():
                                                g_mel_noT, zc, zr, zf = inf_out
                                                g_mel_noT = g_mel_noT.squeeze().T[:, :mel_length]
                                                zc = zc.squeeze().T[:, :mel_length]
                                                zr = zr.squeeze().T[:, :mel_length]
                                                zf = zf.squeeze().T[:, :mel_length]

                                        # create meta folder structure  
                                        file = file_path.split('/')[-1][:-4]
                                        meta_path = os.path.join(meta_dir, dir, file)
                                        if not os.path.exists(meta_path):
                                                os.makedirs(meta_path)

                                        # convert to numpy ndarray and save as file in meta dir
                                        mspec = mspec.T
                                        np.save(os.path.join(meta_path, file + '_mspec.npy'), mspec)
                                        g_mel_noT = g_mel_noT.cpu().detach().numpy()
                                        np.save(os.path.join(meta_path, file + '_mspecOutNoT.npy'), g_mel_noT)
                                        zc = zc.cpu().detach().numpy()
                                        np.save(os.path.join(meta_path, file + '_Zc.npy'), zc)
                                        zr = zr.cpu().detach().numpy()
                                        np.save(os.path.join(meta_path, file + '_Zr.npy'), zr)
                                        zf = zf.cpu().detach().numpy()
                                        np.save(os.path.join(meta_path, file + '_Zf.npy'), zf)



def scatter_plot_regression():
        """
        Plots a scatter plot 
        """
        version = 'UASPEECH_Custom_2'
        meta = 'reference_2'    # meta_0, meta_1, meta_2, meta_3 - and for all of then '_swapped'

        intelligibility = {} 
        with open('/Users/tobi/Data/corpora/' + version + '/intelligibility.csv', mode='r') as handle:
                reader = csv.reader(handle)
                intelligibility = {rows[0]:rows[1] for rows in reader}

        meta_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta + '/pathological'
        path_speaker_codes = {}
        _, dirs, _ = next(os.walk(meta_dir))
        for dir in sorted(dirs):
                speaker = dir
                codes = []
                _, subdirs, _ = next(os.walk(os.path.join(meta_dir, dir)))
                for subdir in sorted(subdirs):
                        _, _, files = next(os.walk(os.path.join(meta_dir, dir, subdir)))
                        for file in sorted(files):
                                if 'Zc_diff_avg.npy' in file:
                                        tmp = np.load(os.path.join(meta_dir, dir, subdir, file))
                                        codes.append(tmp)
                path_speaker_codes[speaker] = codes
        
        meta_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta + '/control'
        control_speaker_codes = {}
        _, dirs, _ = next(os.walk(meta_dir))
        for dir in sorted(dirs):
                speaker = dir
                codes = []
                _, subdirs, _ = next(os.walk(os.path.join(meta_dir, dir)))
                for subdir in sorted(subdirs):
                        _, _, files = next(os.walk(os.path.join(meta_dir, dir, subdir)))
                        for file in sorted(files):
                                if 'Zc_diff_avg.npy' in file:
                                        tmp = np.load(os.path.join(meta_dir, dir, subdir, file))
                                        codes.append(tmp)
                control_speaker_codes[speaker] = codes
        
        # ALL values per speaker - pathological
        all_path_speaker_x = []
        all_path_speaker_y = []
        for speaker in path_speaker_codes:
                for code in path_speaker_codes[speaker]:
                        all_path_speaker_x.append(code)
                        all_path_speaker_y.append(intelligibility[speaker])
                        # all_path_speaker_y.append(1)
        all_path_speaker_y = [float(item) for item in all_path_speaker_y]

        # ALL values per speaker - healthy
        all_heal_speaker_x = []
        all_heal_speaker_y = []
        for speaker in control_speaker_codes:
                for code in control_speaker_codes[speaker]:
                        all_heal_speaker_x.append(code)
                        all_heal_speaker_y.append(intelligibility[speaker])
                        # all_heal_speaker_y.append(0)
        all_heal_speaker_y = [float(item) for item in all_heal_speaker_y]
        
        samples = 760 # max = 700 / 760
        # AVG value per speaker - pathological
        avg_path_speaker_x = []
        avg_path_sepaker_y = []
        for speaker in path_speaker_codes:
                x = np.average(random.sample(path_speaker_codes[speaker], samples))     # look only at randomly chosen X values
                y = intelligibility[speaker]    # intelligibility 1-100%
                # y = 1                                                 # dysarthria == 1
                avg_path_speaker_x.append(x)
                avg_path_sepaker_y.append(y)
        avg_path_sepaker_y = [float(item) for item in avg_path_sepaker_y] 

        # AVG value per speaker - healthy
        avg_heal_speaker_x = []
        avg_heal_speaker_y = []
        for speaker in control_speaker_codes:
                x = np.average(random.sample(control_speaker_codes[speaker], samples))  # look only at randomly chosen X values
                y = intelligibility[speaker]    # intelligibility 1-100%
                # y = 0                                                 # healthy == 0
                avg_heal_speaker_x.append(x)
                avg_heal_speaker_y.append(y)
        avg_heal_speaker_y = [float(item) for item in avg_heal_speaker_y] 

        # AVG - combine pathological and control data
        avg_x = [*avg_path_speaker_x, *avg_heal_speaker_x]
        avg_y = [*avg_path_sepaker_y, *avg_heal_speaker_y]

        # AVG - NO control
        avg_x_p = avg_path_speaker_x
        avg_y_p = avg_path_sepaker_y
        m_p, b_p = np.polyfit(avg_x_p, avg_y_p, 1)
        
        # ALL - combine pathological and control data
        all_x = [*all_path_speaker_x, *all_heal_speaker_x]
        all_y = [*all_path_speaker_y, *all_heal_speaker_y]

        # AVG - linear regression (intelligibility)
        m, b = np.polyfit(avg_x, avg_y, 1)
        pr = scipy.stats.pearsonr(avg_x, avg_y)
        pr_v = pr[0]
        pr_p = pr[1]
        print(f'PEARSON R : {pr}')
        sr = scipy.stats.spearmanr(avg_x, avg_y)
        sr_v = sr[0]
        sr_p = sr[1]
        print(f'SPEARMAN R : {sr}')
        # scatter plot
        nice_scatter(avg_heal_speaker_x, avg_heal_speaker_y, avg_path_speaker_x, avg_path_sepaker_y, samples, avg_x, avg_x_p, m, b, m_p, b_p)



def nice_scatter(x_c, y_c, x_p, y_p, samples, avg_x, avg_x_p, m, b, m_p, b_p):
        """
        Plots a nice scatter plot, for usage in the paper. Specify which meta dir you want to use.
        """
        out_path = '/Users/tobi/Google Drive/study/PhD/research/my_papers/interspeech2022/draft/workbench/scatter'
        times_font = {'fontname':'Times New Roman'}
        font = font_manager.FontProperties(family='Times New Roman', style='normal', size=14)

        plt.scatter(x_p, y_p, s=100, color='red', marker='x', label=f'pathological speaker')
        plt.scatter(x_c, y_c, s=100, facecolors='none', edgecolors='#4FAD5B', label=f'healthy speaker')
        # plt.plot(np.float64(avg_x_p), m_p*np.float64(avg_x_p) + b_p, color='lightgray', linewidth=1.0, label='regression line (only pathological)')
        # plt.plot(np.float64(avg_x), m*np.float64(avg_x) + b, color='darkgray', linewidth=1.0, label='regression line (including healthy)')
        print(f'm: {m}, b: {b}')
        print(f'm_p: {m_p}, b_p: {b_p}')
        plt.plot(np.float64(avg_x), m*np.float64(avg_x) + b, color='darkgray', linewidth=1.0, label='regression line')
        plt.ylim(ymax=103, ymin=-3)
        plt.xlim(xmax=0.035, xmin=0.01)
        plt.xticks(fontsize=14, **times_font)
        plt.yticks(fontsize=14, **times_font)
        plt.legend(edgecolor='black', prop=font)
        plt.ylabel('subjective intelligibility score [%]', fontsize=18, **times_font)
        plt.xlabel('intelligibility index I', fontsize=18, **times_font)
        figure = plt.gcf()
        figure.set_size_inches(8,6)
        plt.savefig(os.path.join(out_path, 'scatter_out.pdf'), bbox_inches='tight')
        plt.show()
                


def dtw_diff_condense():
        """
        4) DTW codes from pathological to healthy reference (gender matched)
        5) calculate aligned absolute square differences between each code (3)
        6) condense aligned absolute square differences to single value (1. avg along value dim, 2. sum along time dim)
        -> stores in metadir
        """

        # meta_0 and reference_1: CF05 / CM04 - instead CM13
        # meta_1 and reference_2 : CF03 / CM05
        # meta_2: CF04 / CM06
        # meta_3 : CF02 / CM08
        # and '_swapped' for each, where F and M reference are swapped
        meta_dir = 'reference_4'        # meta_3 is best performance
        # if '_swapped' just swap the speaker ID below (F <-> M) AND also change 'F' and 'M' below!
        meta_ref_f = 'CF02'
        meta_ref_m = 'CM08'
        version = 'UASPEECH_Custom_3'
        root_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta_dir

        ref_f, ref_m = dict(), dict()   # order: zc, zf, zr, mspec, mspecOutNoT
        ref_dir = '/Users/tobi/Data/corpora/' + version + '/meta/reference'
        ref_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta_dir + '/reference'
        _, dirs, _ = next(os.walk(ref_dir))
        for dir in sorted(dirs):
                # female reference
                if 'F' in dir:  # CHANGE: this to 'M' for '_swapped'
                        _, subdirs, _ = next(os.walk(os.path.join(ref_dir, dir)))
                        for subdir in sorted(subdirs):
                                _, _, files = next(os.walk(os.path.join(ref_dir, dir, subdir)))
                                tmp = []
                                for file in sorted(files):
                                        if '.npy' in file:
                                                tmp.append(np.load(os.path.join(ref_dir, dir, subdir, file)))
                                tid = subdir.split('_')[1:3]
                                fid = '_'.join(tid)
                                ref_f[fid] = tmp
        
                # male reference
                if 'M' in dir:  # CHANGE: this to 'F' for '_swapped'
                        _, subdirs, _ = next(os.walk(os.path.join(ref_dir, dir)))
                        for subdir in sorted(subdirs):
                                _, _, files = next(os.walk(os.path.join(ref_dir, dir, subdir)))
                                tmp = []
                                for file in sorted(files):
                                        if '.npy' in file:
                                                tmp.append(np.load(os.path.join(ref_dir, dir, subdir, file)))
                                tid = subdir.split('_')[1:3]
                                fid = '_'.join(tid)
                                ref_m[fid] = tmp

        # check if all reference files are also in the other folders (disregarding mic)
        ref_path_f = '/Users/tobi/Data/corpora/' + version + '/' + meta_dir + '/reference/' + meta_ref_f
        ref_path_m = '/Users/tobi/Data/corpora/' + version + '/' + meta_dir + '/reference/' + meta_ref_m
        _, subdirs_m, _ = next(os.walk(ref_path_m))
        _, subdirs_f, _ = next(os.walk(ref_path_f))

        m = []
        for subdir_m in sorted(subdirs_m):
                tmp = subdir_m.split('_')[1:3]
                tmp2 = '_'.join(tmp)
                m.append(tmp2)
        f = []
        for subdir_f in sorted(subdirs_f):
                tmp = subdir_f.split('_')[1:3]
                tmp2 = '_'.join(tmp)
                f.append(tmp2)
        for dir in sorted(m):
                if dir not in f:
                        print('missmatch')

        _, dirs, _ = next(os.walk(root_dir))
        for dir in sorted(dirs):
                _, subdirs, _ = next(os.walk(os.path.join(root_dir, dir)))
                for subdir in sorted(subdirs):
                        speaker_files = []
                        _, filedirs, _ = next(os.walk(os.path.join(root_dir, dir, subdir)))
                        for filedir in sorted(filedirs):
                                tmp = filedir.split('_')[1:3]
                                tmp2 = '_'.join(tmp)
                                speaker_files.append(tmp2)
                        
                        for refdir in sorted(m):
                                if refdir not in speaker_files:
                                        print(f'Mistake: {dir}/{subdir}/{refdir} should be there, but is not!')

        # DTW calculations (pathological -to-> gender matched reference)
        _, dirs, _ = next(os.walk(root_dir))
        for dir in sorted(dirs):
                if 'reference' in dir: 
                        continue
                _, subdirs, _ = next(os.walk(os.path.join(root_dir, dir)))

                for subdir in sorted(subdirs):
                        _, filedirs, _ = next(os.walk(os.path.join(root_dir, dir, subdir)))
                        gender = 'M' if 'M' in subdir else 'F'

                        for filedir in sorted(filedirs):
                                _, _, files = next(os.walk(os.path.join(root_dir, dir, subdir, filedir)))
                                tmp = filedir.split('_')[1:3]
                                fid = '_'.join(tmp)
                                # ref order: zc, zf, zr, mspec, mspecOutNoT
                                if 'M' in gender:
                                        ref = ref_m[fid]
                                if 'F' in gender:
                                        ref = ref_f[fid]

                                # ~ mspec
                                mspec_file = [file if file.split('_')[4] == 'mspec.npy' else None for file in sorted(files)]    
                                mspec_file = next(item for item in mspec_file if item is not None)
                                mspec_file_path = mspec_file
                                mspec_file = np.load(os.path.join(root_dir, dir, subdir, filedir, mspec_file))
                                mspec_ref = ref[3]
                                # dtw
                                mspec_dist, mspec_path = dtw(mspec_ref, mspec_file)
                                mspec_dtw = np.empty(shape=(mspec_ref.shape), dtype=np.float32)
                                for i_ref, i_file in mspec_path:
                                        mspec_dtw[:, i_ref] = mspec_file[:, i_file]
                                # abs. sqr. diff. 
                                # mspec_diff = abs(np.square(mspec_ref) - np.square(mspec_dtw))
                                mspec_diff = np.square((mspec_ref - mspec_dtw))
                                # save dtw and diff results as file
                                mspec_dtw_file = mspec_file_path[:-4] + '_dtw.npy'
                                mspec_diff_file = mspec_file_path[:-4] + '_diff.npy'
                                mspec_ref_file = mspec_file_path[:-4] + '_ref.npy'
                                mspec_dtw_file_path = os.path.join(root_dir, dir, subdir, filedir, mspec_dtw_file)
                                mspec_diff_file_path = os.path.join(root_dir, dir, subdir, filedir, mspec_diff_file)
                                mspec_ref_file_path = os.path.join(root_dir, dir, subdir, filedir, mspec_ref_file) 
                                np.save(mspec_dtw_file_path, mspec_dtw)
                                np.save(mspec_diff_file_path, mspec_diff)
                                np.save(mspec_ref_file_path, mspec_ref)

                                # ~ mspecOutNoT
                                mspecOutNoT_file = [file if file.split('_')[4] == 'mspecOutNoT.npy' else None for file in sorted(files)]
                                mspecOutNoT_file = next(item for item in mspecOutNoT_file if item is not None)
                                mspecOutNoT_file_path = mspecOutNoT_file
                                mspecOutNoT_file = np.load(os.path.join(root_dir, dir, subdir, filedir, mspecOutNoT_file))
                                mspecOutNoT_ref = ref[4]
                                # dtw
                                mspecOutNoT_dist, mspecOutNoT_path = dtw(mspecOutNoT_ref, mspecOutNoT_file)
                                mspecOutNoT_dtw = np.empty(shape=(mspecOutNoT_ref.shape), dtype=np.float32)
                                for i_ref, i_file in mspecOutNoT_path:
                                        mspecOutNoT_dtw[:, i_ref] = mspecOutNoT_file[:, i_file]
                                # abs. sqr. diff. 
                                # mspecOutNoT_diff = abs(np.square(mspecOutNoT_ref) - np.square(mspecOutNoT_dtw))
                                mspecOutNoT_diff = np.square((mspecOutNoT_ref - mspecOutNoT_dtw))
                                # save dtw and diff results as file
                                mspecOutNoT_dtw_file = mspecOutNoT_file_path[:-4] + '_dtw.npy'
                                mspecOutNoT_diff_file = mspecOutNoT_file_path[:-4] + '_diff.npy'
                                mspecOutNoT_ref_file = mspecOutNoT_file_path[:-4] + '_ref.npy'
                                mspecOutNoT_dtw_file_path = os.path.join(root_dir, dir, subdir, filedir, mspecOutNoT_dtw_file)
                                mspecOutNoT_diff_file_path = os.path.join(root_dir, dir, subdir, filedir, mspecOutNoT_diff_file)
                                mspecOutNoT_ref_file_path = os.path.join(root_dir, dir, subdir, filedir, mspecOutNoT_ref_file)
                                np.save(mspecOutNoT_dtw_file_path, mspecOutNoT_dtw)
                                np.save(mspecOutNoT_diff_file_path, mspecOutNoT_diff)
                                np.save(mspecOutNoT_ref_file_path, mspecOutNoT_ref)

                                # ~ Zc
                                zc_file = [file if file.split('_')[4] == 'Zc.npy' else None for file in sorted(files)]
                                zc_file = next(item for item in zc_file if item is not None)
                                zc_file_path = zc_file
                                zc_file = np.load(os.path.join(root_dir, dir, subdir, filedir, zc_file))
                                zc_ref = ref[0]
                                # dtw
                                zc_dist, zc_path = dtw(zc_ref, zc_file)
                                zc_dtw = np.empty(shape=(zc_ref.shape), dtype=np.float32)
                                for i_ref, i_file in zc_path:
                                        zc_dtw[:, i_ref] = zc_file[:, i_file]
                                # abs. sqr. diff. 
                                # zc_diff = abs(np.square(zc_ref) - np.square(zc_dtw))
                                zc_diff = np.square((zc_ref - zc_dtw))
                                # zc_diff = scipy.signal.correlate(zc_ref, zc_dtw)      # cross-correlation
                                # condense (i.e. average) code
                                zc_diff_avg_y = np.mean(zc_diff, axis=0)
                                zc_diff_avg = np.mean(zc_diff_avg_y, axis=0)
                                # zc_diff_avg_y = np.mean(zc_diff, axis=0)                      # cross-correlation
                                # zc_diff_avg = np.mean(zc_diff_avg_y, axis=0)          # cross-correlation

                                # save dtw and diff results as file
                                zc_dtw_file = zc_file_path[:-4] + '_dtw.npy'
                                zc_diff_file = zc_file_path[:-4] + '_diff.npy'
                                zc_ref_file = zc_file_path[:-4] + '_ref.npy'
                                zc_diff_avg_file = zc_file_path[:-4] + '_diff_avg.npy'
                                zc_dtw_file_path = os.path.join(root_dir, dir, subdir, filedir, zc_dtw_file)
                                zc_diff_file_path = os.path.join(root_dir, dir, subdir, filedir, zc_diff_file)
                                zc_ref_file_path = os.path.join(root_dir, dir, subdir, filedir, zc_ref_file)
                                zc_diff_avg_file_path = os.path.join(root_dir, dir, subdir, filedir, zc_diff_avg_file)
                                np.save(zc_dtw_file_path, zc_dtw)
                                np.save(zc_diff_file_path, zc_diff)
                                np.save(zc_ref_file_path, zc_ref)
                                np.save(zc_diff_avg_file_path, zc_diff_avg)

                                # ~ Zf
                                zf_file = [file if file.split('_')[4] == 'Zf.npy' else None for file in sorted(files)]
                                zf_file = next(item for item in zf_file if item is not None)
                                zf_file_path = zf_file
                                zf_file = np.load(os.path.join(root_dir, dir, subdir, filedir, zf_file))
                                zf_ref = ref[1]
                                # dtw
                                zf_dist, zf_path = dtw(zf_ref, zf_file)
                                zf_dtw = np.empty(shape=(zf_ref.shape), dtype=np.float32)
                                for i_ref, i_file in zf_path:
                                        zf_dtw[:, i_ref] = zf_file[:, i_file]
                                # abs. sqr. diff. 
                                # zf_diff = abs(np.square(zf_ref) - np.square(zf_dtw))
                                zf_diff = np.square((zf_ref - zf_dtw))
                                # condense (i.e. average) code
                                zf_diff_avg_y = np.mean(zf_diff, axis=0)
                                zf_diff_avg = np.mean(zf_diff_avg_y, axis=0)
                                # save dtw and diff results as file
                                zf_dtw_file = zf_file_path[:-4] + '_dtw.npy'
                                zf_diff_file = zf_file_path[:-4] + '_diff.npy'
                                zf_ref_file = zf_file_path[:-4] + '_ref.npy'
                                zf_diff_avg_file = zf_file_path[:-4] + '_diff_avg.npy'
                                zf_dtw_file_path = os.path.join(root_dir, dir, subdir, filedir, zf_dtw_file)
                                zf_diff_file_path = os.path.join(root_dir, dir, subdir, filedir, zf_diff_file)
                                zf_ref_file_path = os.path.join(root_dir, dir, subdir, filedir, zf_ref_file)
                                zf_diff_avg_file_path = os.path.join(root_dir, dir, subdir, filedir, zf_diff_avg_file)
                                np.save(zf_dtw_file_path, zf_dtw)
                                np.save(zf_diff_file_path, zf_diff)
                                np.save(zf_ref_file_path, zf_ref)
                                np.save(zf_diff_avg_file_path, zf_diff_avg)

                                # ~ Zr
                                zr_file = [file if file.split('_')[4] == 'Zr.npy' else None for file in sorted(files)]
                                zr_file = next(item for item in zr_file if item is not None)
                                zr_file_path = zr_file
                                zr_file = np.load(os.path.join(root_dir, dir, subdir, filedir, zr_file))
                                zr_ref = ref[2]
                                # dtw
                                zr_dist, zr_path = dtw(zr_ref, zr_file)
                                zr_dtw = np.empty(shape=(zr_ref.shape), dtype=np.float32)
                                for i_ref, i_file in zr_path:
                                        zr_dtw[:, i_ref] = zr_file[:, i_file]
                                # abs. sqr. diff. 
                                # zr_diff = abs(np.square(zr_ref) - np.square(zr_dtw))
                                zr_diff = np.square((zr_ref - zr_dtw))
                                # condense (i.e. average) code
                                zr_diff_avg_y = np.mean(zr_diff, axis=0)
                                zr_diff_avg = np.mean(zr_diff_avg_y, axis=0)
                                # save dtw and diff results as file
                                zr_dtw_file = zr_file_path[:-4] + '_dtw.npy'
                                zr_diff_file = zr_file_path[:-4] + '_diff.npy'
                                zr_ref_file = zr_file_path[:-4] + '_ref.npy'
                                zr_diff_avg_file = zr_file_path[:-4] + '_diff_avg.npy'
                                zr_dtw_file_path = os.path.join(root_dir, dir, subdir, filedir, zr_dtw_file)
                                zr_diff_file_path = os.path.join(root_dir, dir, subdir, filedir, zr_diff_file)
                                zr_ref_file_path = os.path.join(root_dir, dir, subdir, filedir, zr_ref_file)
                                zr_diff_avg_file_path = os.path.join(root_dir, dir, subdir, filedir, zr_diff_avg_file)
                                np.save(zr_dtw_file_path, zr_dtw)
                                np.save(zr_diff_file_path, zr_diff)
                                np.save(zr_ref_file_path, zr_ref)
                                np.save(zr_diff_avg_file_path, zr_diff_avg)



def reset_meta_dir():
        """
        Prepares the meta folders for the experiment with 3 different sets of (gender matched) reference speakers:
                - delete all the files that contain the following words from the pathological and control folders:
                        - diff, dtw, ref 
                        - after deletion, only 5 files should be in the folders: 3 codes, mspec, mspecOutNoT
                - exchange speakers in the reference dir with 2 different ones from the control dir
        """
        
        meta = 'meta_0_swapped'
        root_dir = '/Users/tobi/Data/corpora/UASPEECH_Custom/' + meta

        _, dirs, _ = next(os.walk(root_dir))
        for dir in sorted(dirs):
                if 'reference' in dir:
                        continue
                _, subdirs, _ = next(os.walk(os.path.join(root_dir, dir)))
                for subdir in sorted(subdirs):
                        _, filedirs, _ = next(os.walk(os.path.join(root_dir, dir, subdir)))
                        for filedir in sorted(filedirs):
                                _, _, files = next(os.walk(os.path.join(root_dir, dir, subdir, filedir)))
                                for file in sorted(files):
                                        if 'npy' in file:
                                                if 'diff' in file:
                                                        os.remove(os.path.join(root_dir, dir, subdir, filedir, file))
                                                if 'dtw' in file:
                                                        os.remove(os.path.join(root_dir, dir, subdir, filedir, file))
                                                if 'ref' in file:
                                                        os.remove(os.path.join(root_dir, dir, subdir, filedir, file))



def n_utterances_t_times(n, T):
        """
        Randomly picks n utterances per speaker (same ones for each!) and does the correlation for them t times.
        The results are stored in a .csv file. 
        Choosing meta_3 (since best performance) as root_dir. 
        """
        
        code = 'Zf'     # Zc, Zf, Zr
        pathological_only = False
        without_UW_utterances = False
        version = 'UASPEECH_Custom_2'
        meta = 'reference_2'    # meta_0, meta_1, meta_2, meta_3 - and for all of then '_swapped'
        # root_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta
        # target_dir = '/Users/tobi/Data/corpora/' + version + '/n_utterances/10'

        intelligibility = {} 
        with open('/Users/tobi/Data/corpora/' + version + '/intelligibility.csv', mode='r') as handle:
                reader = csv.reader(handle)
                intelligibility = {rows[0]:rows[1] for rows in reader}

        meta_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta + '/pathological'
        path_speaker_codes = {}
        _, dirs, _ = next(os.walk(meta_dir))
        for dir in sorted(dirs):
                speaker = dir
                codes = []
                _, subdirs, _ = next(os.walk(os.path.join(meta_dir, dir)))
                for subdir in sorted(subdirs):
                        _, _, files = next(os.walk(os.path.join(meta_dir, dir, subdir)))
                        for file in sorted(files):
                                if (code + '_diff_avg.npy') in file:
                                        if without_UW_utterances:
                                                if '_UW' not in file:
                                                        tmp = np.load(os.path.join(meta_dir, dir, subdir, file))
                                                        codes.append(tmp)
                                        else:
                                                tmp = np.load(os.path.join(meta_dir, dir, subdir, file))
                                                codes.append(tmp)
                path_speaker_codes[speaker] = codes
        
        meta_dir = '/Users/tobi/Data/corpora/' + version + '/' + meta + '/control'
        control_speaker_codes = {}
        _, dirs, _ = next(os.walk(meta_dir))
        for dir in sorted(dirs):
                speaker = dir
                codes = []
                _, subdirs, _ = next(os.walk(os.path.join(meta_dir, dir)))
                for subdir in sorted(subdirs):
                        _, _, files = next(os.walk(os.path.join(meta_dir, dir, subdir)))
                        for file in sorted(files):
                                if (code + '_diff_avg.npy') in file:
                                        if without_UW_utterances:
                                                if '_UW' not in file:
                                                        tmp = np.load(os.path.join(meta_dir, dir, subdir, file))
                                                        codes.append(tmp)
                                        else:
                                                tmp = np.load(os.path.join(meta_dir, dir, subdir, file))
                                                codes.append(tmp)
                control_speaker_codes[speaker] = codes

        for speaker in control_speaker_codes:
                print(len(control_speaker_codes[speaker]))
        
        all_pr = []
        avg_pr = 0
        worst_pr_p = 0
        all_sr = []
        avg_sr = 0
        worst_sr_p = 0

        utt_count = 463 if without_UW_utterances else 760
        for t in range(0, T):
                # select n random indices, to choose the SAME utterances for each speaker (max is 700 / 760)
                idx = np.random.choice(utt_count, n, replace=False)

                # AVG value per speaker - pathological
                avg_path_speaker_x = []
                avg_path_speaker_y = []
                for speaker in path_speaker_codes:
                        tmp = []
                        for i in idx:
                                tmp.append(path_speaker_codes[speaker][i])
                        x = np.average(tmp)
                        y = intelligibility[speaker]    # intelligibility 1-100%
                        # y = 1                                                 # dysarthria == 1
                        avg_path_speaker_x.append(x)
                        avg_path_speaker_y.append(y)
                avg_path_speaker_y = [float(item) for item in avg_path_speaker_y] 

                # AVG value per speaker - healthy
                avg_heal_speaker_x = []
                avg_heal_speaker_y = []
                for speaker in control_speaker_codes:
                        tmp = []
                        for i in idx:
                                tmp.append(control_speaker_codes[speaker][i])
                        x = np.average(tmp)
                        y = intelligibility[speaker]    # intelligibility 1-100%
                        # y = 0                                                 # healthy == 0
                        avg_heal_speaker_x.append(x)
                        avg_heal_speaker_y.append(y)
                avg_heal_speaker_y = [float(item) for item in avg_heal_speaker_y] 

                # AVG - combine pathological and control data
                avg_x = [*avg_path_speaker_x, *avg_heal_speaker_x]
                avg_y = [*avg_path_speaker_y, *avg_heal_speaker_y]
                
                # not looking at control speakers decreases the correlation
                if pathological_only:
                        avg_x = avg_path_speaker_x
                        avg_y = avg_path_speaker_y

                # AVG - linear regression (intelligibility)
                m, b = np.polyfit(avg_x, avg_y, 1)
                pr = scipy.stats.pearsonr(avg_x, avg_y)
                pr_v = pr[0]
                pr_p = pr[1]
                # print(f'PEARSON R : {pr}')
                sr = scipy.stats.spearmanr(avg_x, avg_y)
                sr_v = sr[0]
                sr_p = sr[1]
                # print(f'SPEARMAN R : {sr}')

                # update
                if pr_p > worst_pr_p:
                        worst_pr_p = pr_p
                if sr_p > worst_sr_p:
                        worst_sr_p = sr_p
                all_pr.append(pr_v)
                all_sr.append(sr_v)
        
        
        if T != 1:
                # avg_pr = np.average(all_pr)
                # avg_sr = np.average(all_sr)
                print('---------------------------')
                print(f'PEARSON mean: {statistics.mean(all_pr)}')
                print(f'PEARSON stdev: {statistics.stdev(all_pr)}')
                print(f'PEARSON worst p-value: {worst_pr_p}')
                print('---------------------------')
                print(f'SPEARMAN mean: {statistics.mean(all_sr)}')
                print(f'SPEARMAN stdev: {statistics.stdev(all_sr)}')
                print(f'SPEARMAN worst p-value: {worst_sr_p}')
        else:
                print(f'PEARSON R: {all_pr}')
                print(f'PEARSON p-value: {worst_pr_p}')
                print(f'SPEARMAN R: {all_sr}')
                print(f'SPEARMAN p-value: {worst_sr_p}')


def plot_everything(plot_dir):
        """
        Plots everything that is contained in the plot_dir (specific meta dir of a patient and specific file).
        """
        # of files to plot
        file_id = plot_dir.split('/')[-1] 

        # mspec
        mspec_file = np.load(os.path.join(plot_dir, file_id + '_mspec.npy'))
        mspec_ref = np.load(os.path.join(plot_dir, file_id + '_mspec_ref.npy')) 
        mspec_dtw = np.load(os.path.join(plot_dir, file_id + '_mspec_dtw.npy'))
        mspec_diff = np.load(os.path.join(plot_dir, file_id + '_mspec_diff.npy'))

        # mspecOutNoT
        mspecOutNoT_file = np.load(os.path.join(plot_dir, file_id + '_mspecOutNoT.npy'))
        mspecOutNoT_ref = np.load(os.path.join(plot_dir, file_id + '_mspecOutNoT_ref.npy'))
        mspecOutNoT_dtw = np.load(os.path.join(plot_dir, file_id + '_mspecOutNoT_dtw.npy'))
        mspecOutNoT_diff = np.load(os.path.join(plot_dir, file_id + '_mspecOutNoT_diff.npy'))

        # Zc
        zc_file = np.load(os.path.join(plot_dir, file_id + '_zc.npy'))
        zc_ref = np.load(os.path.join(plot_dir, file_id + '_zc_ref.npy'))
        zc_dtw = np.load(os.path.join(plot_dir, file_id + '_zc_dtw.npy'))
        zc_diff = np.load(os.path.join(plot_dir, file_id + '_zc_diff.npy'))

        # Zf
        zf_file = np.load(os.path.join(plot_dir, file_id + '_zf.npy'))
        zf_ref = np.load(os.path.join(plot_dir, file_id + '_zf_ref.npy'))
        zf_dtw = np.load(os.path.join(plot_dir, file_id + '_zf_dtw.npy'))
        zf_diff = np.load(os.path.join(plot_dir, file_id + '_zf_diff.npy'))

        # Zr
        zr_file = np.load(os.path.join(plot_dir, file_id + '_zr.npy'))
        zr_ref = np.load(os.path.join(plot_dir, file_id + '_zr_ref.npy'))
        zr_dtw = np.load(os.path.join(plot_dir, file_id + '_zr_dtw.npy'))
        zr_diff = np.load(os.path.join(plot_dir, file_id + '_zr_diff.npy'))

        with torch.no_grad():
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,4, constrained_layout=True, sharey='row')
                # mspec
                ax1[0].set_title('1: mspec (P)', color='grey')
                # im1 = ax1[0].imshow(mspec_file, aspect='auto')
                librosa.display.specshow(mspec_file, x_axis='frames', y_axis='hz', ax=ax1[0])
                ax1[1].set_title('2: mspec (H)')
                # im1 = ax1[1].imshow(mspec_ref, aspect='auto')
                librosa.display.specshow(mspec_ref, x_axis='frames', y_axis='hz', ax=ax1[1])
                ax1[2].set_title('3: mspec DTW 1->2')
                # im1 = ax1[2].imshow(mspec_dtw, aspect='auto')
                librosa.display.specshow(mspec_dtw, x_axis='frames', y_axis='hz', ax=ax1[2])
                ax1[3].set_title('4: mspec DIFF 2 & 3')
                # im1 = ax1[3].imshow(mspec_diff, aspect='auto')
                librosa.display.specshow(mspec_diff, x_axis='frames', y_axis='hz', ax=ax1[3])

                # mspecOutNoT
                ax2[0].set_title('mspecOutNoT (P)', color='grey')
                # im2 = ax2[0].imshow(mspecOutNoT_file, aspect='auto')
                librosa.display.specshow(mspecOutNoT_file, x_axis='frames', y_axis='hz', ax=ax2[0])
                ax2[1].set_title('mspecOutNoT (H)')
                # im2 = ax2[1].imshow(mspecOutNoT_ref, aspect='auto')
                librosa.display.specshow(mspecOutNoT_ref, x_axis='frames', y_axis='hz', ax=ax2[1])
                ax2[2].set_title('mspecOutNoT DTW (P)')
                # im2 = ax2[2].imshow(mspecOutNoT_dtw, aspect='auto')
                librosa.display.specshow(mspecOutNoT_dtw, x_axis='frames', y_axis='hz', ax=ax2[2])
                ax2[3].set_title('mspecOutNoT DIFF')
                # im2 = ax2[3].imshow(mspecOutNoT_diff, aspect='auto')
                librosa.display.specshow(mspecOutNoT_diff, x_axis='frames', y_axis='hz', ax=ax2[3])

                # Zc
                ax3[0].set_title('Zc (P)', color='grey')
                im3 = ax3[0].imshow(zc_file, aspect='auto', origin='lower')
                ax3[1].set_title('Zc (H)')
                im3 = ax3[1].imshow(zc_ref, aspect='auto', origin='lower')
                ax3[2].set_title('Zc DTW (P)')
                im3 = ax3[2].imshow(zc_dtw, aspect='auto', origin='lower')
                ax3[3].set_title('Zc DIFF')
                im3 = ax3[3].imshow(zc_diff, aspect='auto', origin='lower')

                # Zf
                ax4[0].set_title('Zf (P)', color='grey')
                im4 = ax4[0].imshow(zf_file, aspect='auto', origin='lower')
                ax4[1].set_title('Zf (H)')
                im4 = ax4[1].imshow(zf_ref, aspect='auto', origin='lower')
                ax4[2].set_title('Zf DTW (P)')
                im4 = ax4[2].imshow(zf_dtw, aspect='auto', origin='lower')
                ax4[3].set_title('Zf DIFF')
                im4 = ax4[3].imshow(zf_diff, aspect='auto', origin='lower')

                # Zr
                ax5[0].set_title('Zr (P)', color='grey')
                im5 = ax5[0].imshow(zr_file, aspect='auto', origin='lower')
                ax5[1].set_title('Zr (H)')
                im5 = ax5[1].imshow(zr_ref, aspect='auto', origin='lower')
                ax5[2].set_title('Zr DTW (P)')
                im5 = ax5[2].imshow(zr_dtw, aspect='auto', origin='lower')
                ax5[3].set_title('Zr DIFF')
                im5 = ax5[3].imshow(zr_diff, aspect='auto', origin='lower')

                plt.show()
                plt.close(fig)



if __name__ == '__main__':
        # ~ Correlation testing (perform functions below in the listed order)
        # 1) prepare a custom UASpeech corpus, that will be used in all the following steps
        # create_UASpeech_custom_v2()
        # 2) cut 15% of the audio durations at the beginning and end, then perform VAD
        # cut_vad(0.15, 0.0)
        # 3) perform inference with the previously trained SpeechSplit model
        # inference()
        # 4) to test across the 4 available reference speaker pairs
        # manually create the 4 reference dir's and copy files from inference dir in: control, reference, pathological dirs
        # 5) calculate all the required metrics
        # dtw_diff_condense()
        # 6) scatter plot including regression line
        # scatter_plot_regression()
        # 7) get correlation results for different amount of data used
        # n_utterances_t_times(760,1)   # 20,1000 and 760,1 -> change "pathological_only"

        # ~ Some auxilary functions (to help with intuition about the data/representations)
        # reset_meta_dir()
        # plot_everything('/Users/tobi/Data/corpora/UASPEECH_Custom_2/reference_2/pathological/F02/F02_B1_C4_M3')
        # plot_everything('/Users/tobi/Data/corpora/UASPEECH_Custom/meta_0/pathological/F02/F02_B1_C2_M2')

        print('main done.')

