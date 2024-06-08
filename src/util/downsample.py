import os
import torchaudio
from glob import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, help='directory for high sampling rate audio files')
parser.add_argument('--tgt_dir', type=str, help='direcotry to save downsampled audio files')
parser.add_argument('--sr', type=int, default=8000, help='desired downsampling rate')
args = parser.parse_args()


if not os.path.exists(args.tgt_dir):
    os.makedirs(args.tgt_dir)

transform = torchaudio.transforms.Resample(16000, args.sr)
transform_up = torchaudio.transforms.Resample(args.sr, 16000)

wav_path = sorted(glob(args.src_dir + '/*.wav'))

for i,path in enumerate(tqdm(wav_path)):
    
    w, sr = torchaudio.load(path, normalize=True)
    w_ = transform(w)
    w_ = transform_up(w_)
    path_8k = os.path.join(args.tgt_dir, path.split("/")[-1])
    torchaudio.save(path_8k,w_,16000)
