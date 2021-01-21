#!/bin/bash

. ./cmd.sh
. ./path.sh

n=20
set -eu
rm -rf data/fbank/enroll && mkdir -p data/fbank/enroll && cp -r data/enroll data/fbank

steps/make_fbank.sh --nj $n --cmd "$train_cmd" data/fbank/enroll
cp data/mfcc/enroll/vad.scp data/fbank/enroll/vad.scp

