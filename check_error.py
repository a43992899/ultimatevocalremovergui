# load two .wav, and check if wav1 - wav2 is close to 0, and the max/min/mean of the difference

import numpy as np
import torchaudio

def check_error(wav1, wav2):
    wav1 = torchaudio.load(wav1)[0].numpy()
    wav2 = torchaudio.load(wav2)[0].numpy()
    diff = wav1 - wav2
    print("Max diff: ", np.max(diff))
    print("Min diff: ", np.min(diff))
    print("Mean diff: ", np.mean(diff))
    print("Close to 0: ", np.allclose(wav1, wav2))

if __name__ == "__main__":
    wav1 = "/scratch/buildlam/codeclm/ultimatevocalremovergui/data/test_output/_1_all-171130.0-180870.0_(Vocals).wav"
    wav2 = "/scratch/buildlam/codeclm/ultimatevocalremovergui/data/test_output/1_all-171130.0-180870.0_(Vocals).wav"
    check_error(wav1, wav2)
