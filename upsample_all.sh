#!/bin/bash

mkdir -p output

nets=`ls models`
images=`ls images`

for image in $images
do
    im=${image%.*}
    for net in $nets
    do
        nt=${net%.*}    
        new_im=${im}_${nt}.jpg
        python3 upsample.py images/${image} output/${new_im} models/${net}
    done
done
