#!/bin/bash

. ./cmd.sh
. ./path.sh

n=10 # parallel jobs
xvector_dim=512
exp=exp/xvector_tdnn_dim${xvector_dim}

set -eu

stage=0
####### BOOKMARK: basic preperation ######

# corpus and trans directory
thchs=/home/nairuiqian/Projects/THCHS30
musan=/home/nairuiqian/Projects/MUSAN/musan
rirs_noises=/home/nairuiqian/Projects/RIRS_NOISES/RIRS_NOISES

trials=data/test/trials

if [ $stage -le 0 ]; then
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
if [ $stage -le 1 ]; then
    # produce MFCC feature with energy and its vad in data/mfcc/{train,enroll,test}
    [ -d data/mfcc ] || mkdir -p data/mfcc 
    rm -rf data/mfcc{train,test,enroll} &&  cp -r data/{train,enroll,test} data/mfcc
    for x in train enroll test; do
        steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $n --cmd "$train_cmd" \
            data/${x} data/mfcc/${x}
        sid/compute_vad_decision.sh --nj $n --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
    done
fi

###### BOOKMARK: data augmentation ######
if [ $stage -le 2 ]; then
    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, ${rirs_noises}/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, ${rirs_noises}/simulated_rirs/mediumroom/rir_list")
    # Make a reverberated version of thch-30
    rm -rf data/train_reverb && mkdir -p data/train_reverb
    python local/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        data/train data/train_reverb $(dirname ${rirs_noises})
    
    for x in train enroll test; do
        cp data/mfcc/${x}/vad.scp data/${x}
    done
    
    cp data/mfcc/train/vad.scp data/train_reverb
    utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
    rm -rf data/train_reverb
    mv data/train_reverb.new data/train_reverb
    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    rm -rf data/musan
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

###### BOOKMARK: feature extracrion for augmented data ######
if [ $stage -le 3 ]; then
    rm -rf data/mfcc/train_aug && cp -r data/train_aug data/mfcc
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 6 --cmd "${train_cmd}" \
        data/train_aug  data/mfcc/train_aug
fi

###### BOOKMARK: examples preperation for xvector trainning ######
if [ $stage -le 4 ]; then
    # Combine the clean and augmented train list, making the list size 4 times larger.
    rm -rf data/train_combined
    utils/combine_data.sh data/train_combined data/train_aug data/train
    utils/fix_data_dir.sh data/train_combined

    # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
    # wasteful, as it roughly doubles the amount of training data on disk.  After
    # creating training examples, this can be removed.
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $n --cmd "$train_cmd" \
        data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
    utils/fix_data_dir.sh data/train_combined_no_sil

    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 2s (200 frames) per utterance.
    min_len=200
    mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
    awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
    utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
    mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
    utils/fix_data_dir.sh data/train_combined_no_sil
    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    min_num_utts=8
    awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
    awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
    mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
    utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk

    utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
    mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames

    # Now we're ready to create training examples.
    utils/fix_data_dir.sh data/train_combined_no_sil

fi

###### BOOKMARK: xvector trainning ######
if [ $stage -le 5 ]; then
    local/nnet3/xvector/run_xvector.sh --stage 0 --train-stage -1 \
        --data data/train_combined_no_sil --nnet-dir $exp \
        --egs-dir ${exp}/egs
fi

###### BOOKMARK: xvector extraction ######
if [ $stage -le 6 ]; then
    for x in train_combined enroll test; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" --nj $n \
            $exp data/${x} \
            "${exp}/xvectors_${x}"
    done
   
fi
###### BOOKMARK: cosine scoring ######
if [ $stage -le 7 ]; then
    # basic cosine scoring on x-vectors
    local/cosine_scoring.sh data/mfcc/enroll data/mfcc/test \
        $exp/xvectors_enroll $exp/xvectors_test $trials $exp/scores

    # cosine scoring after reducing the x-vector dim with LDA
    local/lda_scoring.sh data/train_combined data/mfcc/enroll data/mfcc/test \
        $exp/xvectors_train_combined $exp/xvectors_enroll $exp/xvectors_test $trials $exp/scores

    # cosine scoring after reducing the x-vector dim with PLDA
    local/plda_scoring.sh data/train_combined data/mfcc/enroll data/mfcc/test \
        $exp/xvectors_train_combined $exp/xvectors_enroll $exp/xvectors_test $trials $exp/scores

    # print eer
    for i in cosine lda plda; do
        eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
        printf "%15s %5.2f \n" "$i eer:" $eer
    done
fi

exit 0
