#!/bin/bash

. ./cmd.sh
. ./path.sh

n=10 # parallel jobs
xvector_dim=400
exp=exp/xvector_tdnn_dim${xvector_dim}

set -eu

stage=2
####### BOOKMARK: basic preperation ######

# corpus and trans directory
thchs=/home/nairuiqian/Projects/THCHS30/
musan=/home/nairuiqian/Projects/MUSAN/musan
rirs_noises=/home/nairuiqian/Projects/RIRS_NOISES/RIRS_NOISES

trials=data/test/trials

if [ $stage -eq 0 ]; then
    # generate text, wav.scp, utt2pk, spk2utt in data/{train,test}
    local/thchs-30_data_prep.sh $thchs/data_thchs30
    # randomly select 1000 utts from data/test as enrollment in data/enroll
    # using rest utts in data/test for test
    utils/subset_data_dir.sh data/test 1000 data/enroll
    utils/filter_scp.pl --exclude data/enroll/wav.scp data/test/wav.scp > data/test/wav.scp.rest
    mv data/test/wav.scp.rest data/test/wav.scp
    utils/fix_data_dir.sh data/test

    # prepare trials in data/test
    local/prepare_trials.py data/enroll data/test
fi

###### BOOKMARK: feature extraction ######
if [ $stage -eq 1 ]; then
    # produce MFCC feature with energy and its vad in data/mfcc/{train,enroll,test}
    rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,enroll,test} data/mfcc
    for x in train enroll test; do
        steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x
        sid/compute_vad_decision.sh --nj $n --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
    done
fi

###### BOOKMARK: data augmentation ######
if [ $stage -eq 2 ]; then
    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, ${rirs_noises}/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, ${rirs_noises}/simulated_rirs/mediumroom/rir_list")
    # Make a reverberated version of thch-30
    python steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        data/train data/train_reverb
    
    cp data/mfcc/train/vad.scp data/train
    cp data/mfcc/train/vad.scp data/train_reverb
    utils/copy_data_dir.sh --utt-shufix "-reverb" data/train_reverb data/train_reverb.new
    rm -rf data/train_reverb
    mv data/train_reverb.new data/train_reverb

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    local/make_musan.sh ${musan} data
    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
        utils/data/get_utt2dur.sh data/musan_${name}
        mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done

    # Augment with musan_noise
    python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
    # Augment with musan_music
    python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
    # Augment with musan_speech
    python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

    # Combine reverb, noise, music, and babble into one directory.
    utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi

    