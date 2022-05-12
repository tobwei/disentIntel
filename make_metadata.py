import os
import pickle
import numpy as np

# spmel -> only data, no metadata
rootDir = '/Users/tobi/Data/corpora/CommonSpeakerSplit/en/spmel'
dirName, subDirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# SET THIS: (determines how many of the validation files are used in the val.pkl for actual validation)
valFileCountPerSpeaker = 1

numSpeakers = 14212
maxMelLength = 0    # 661 (de), 547 (en) -> max_pad_len for validation files in hyperparameters
trainSpeakers = []   # 3860 (de), 14212 (en)
valSpeakers = []    # 3860 (de), 14212 (en)
onehot = 0

for speaker in sorted(subDirList):
    print('Processing speaker: %s' % speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
    currentValSpeakerCount = 0

    # add different data to train and val metadata
    for fileName in sorted(fileList):
        if 'train' in fileName:
            # speaker (folder name)
            trainUtterances = []
            trainUtterances.append(speaker)
            # speaker-embedding (one-hot)
            spkid = np.zeros((numSpeakers,), dtype=np.float32)
            spkid[onehot] = 1.0
            trainUtterances.append(spkid)
            # add file list, and add to final list, which will be written to file train.pkl
            trainUtterances.append(os.path.join(speaker, fileName))
            trainSpeakers.append(trainUtterances)
        if 'val' in fileName:
            if currentValSpeakerCount >= valFileCountPerSpeaker:
                continue
            # speaker (folder name)
            valStructure = []
            valStructure.append(speaker)
            # speaker-embedding (one-hot)
            spkid = np.zeros((1,numSpeakers), dtype=np.float32)
            spkid[0,onehot] = 1.0
            valStructure.append(spkid)
            # validation set structure (from github question): [Speaker_Name , One-hot , [Mel, normed-F0, length, filename*]] *with no .npy ?! (also see here: https://github.com/auspicious3000/SpeechSplit/issues/11)
            valData = []
            # melspec
            melDir = rootDir 
            mel = np.load(os.path.join(rootDir, speaker, fileName))
            valData.append(mel) 
            # normed-f0
            raptf0Dir = '/Users/tobi/Data/corpora/CommonSpeakerSplit/en/raptf0'
            raptf0 = np.load(os.path.join(raptf0Dir, speaker, fileName))
            valData.append(raptf0)
            # length
            length = mel.shape[0]
            if length >= maxMelLength:
                maxMelLength = length
            valData.append(length)
            # filename, with no file-extension
            valData.append(fileName[:-4])
            # add valData, and add to final list, which will be written to file val.pkl
            valStructure.append(valData)
            valSpeakers.append(valStructure)
            currentValSpeakerCount += 1
    # next iteration == next speaker (i.e. folder)
    onehot += 1 
print('Maximum mel-length was: ' + str(maxMelLength))

with open(os.path.join(rootDir[:-6], 'train.pkl'), 'wb') as handle:
    pickle.dump(trainSpeakers, handle)    

with open(os.path.join(rootDir[:-6], 'val.pkl'), 'wb') as handle:   # originally called demo.pkl
    pickle.dump(valSpeakers, handle)    

# just for debugging : set debug point here to check structure of pickle file
# demo = pickle.load(open('/Users/tobi/Data/corpora/CommonSpeakerSplit/en/val.pkl', 'rb'))
# print('done.')
