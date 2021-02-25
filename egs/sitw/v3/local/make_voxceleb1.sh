
if [ -f wav.scp ]; then rm wav.scp; fi
if [ -f utt2spk ]; then rm utt2spk; fi

for i in $1/voxceleb1_wav/*/*/*.wav; do
  # spk=`echo $i | cut -d '/' -f 2`
  spk=`echo $i | awk -F "/" '{print $(NF-2)}'`
  utt=`echo $i | awk -F "/" '{print $NF}'`
  utt=${utt%.wav*}
  path=`realpath $i`

  echo "${spk}_$utt $spk" >> utt2spk
  echo "${spk}_$utt $path" >> wav.scp 
done

sort utt2spk -o utt2spk
sort wav.scp -o wav.scp