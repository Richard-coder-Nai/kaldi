#!/bin/bash
. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

voxceleb1_root=/work102/lilt/database/VoxCeleb/voxceleb1
sitw_root=/work102/lilt/database/SITW
nnet_dir=exp/xvector_nnet_1a
musan_root=/work102/lilt/database/musan
rirs_root=/work102/lilt/database/RIRS_NOISES

sitw_dev_trials_core=data/sitw_dev_test/trials/core-core.lst
sitw_eval_trials_core=data/sitw_eval_test/trials/core-core.lst

stage=4

if [ $stage -eq 0 ];then
    # Prepare the VoxCeleb1 dataset and remove the speakers 
    # that ovelap between VoxCeleb1 and SITW.
    # local/make_voxceleb1.sh ${voxceleb1_root} data/voxceleb1
    if ! [ -f data/voxceleb1/voxceleb1_sitw_overlap.txt ];then
      wget -O data/voxceleb1/voxceleb1_sitw_overlap.txt http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt
      python local/trans_overlap_txt.py --meta-csv-file ${voxceleb1_root}/vox1_meta.csv  --overlap-txt-file data/voxceleb1/voxceleb1_sitw_overlap.txt
    fi
    rm -rf data/overlap
    mkdir data/overlap
    utils/subset_data_dir.sh --spk-list data/voxceleb1/voxceleb1_sitw_overlap.txt \
      data/voxceleb1 data/overlap
    
    cp -r data/voxceleb1 data/train
    utils/filter_scp.pl --exclude data/overlap/wav.scp data/voxceleb1/wav.scp > data/train/wav.scp.rest
    mv data/train/wav.scp.rest data/train/wav.scp
    utils/fix_data_dir.sh data/train
    # We will train only on VoxCeleb1.
    # Prepare SITW, which is our test set.
    local/make_sitw.sh $sitw_root data
fi

if [ $stage -eq 1 ];then
    # for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test train; do
    for name in train; do

         steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
           data/${name} exp/make_mfcc $mfccdir
         utils/fix_data_dir.sh data/${name}
         sid/compute_vad_decision.sh --nj 80 --cmd "$train_cmd" \
           data/${name} exp/make_vad $vaddir
         utils/fix_data_dir.sh data/${name}
    done

fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -eq 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur
  if [ ! -d "RIRS_NOISES" ];then
    ln -s $rirs_root ./
  fi
  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan_root data

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

if [ $stage -eq 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/train_aug 100000 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi
# Now we prepare the features to generate examples for xvector training.
if [ $stage -eq 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 80 --cmd "$train_cmd"  data/train_combined data/train_combined_no_sil exp/train_combined_no_sil 
  utils/fix_data_dir.sh data/train_combined_no_sil 
fi



if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
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

# Stages 6 through 8 are handled in run_xvector.sh
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/train_combined_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs



