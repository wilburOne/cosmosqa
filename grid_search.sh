#!/bin/bash

task="commonsenseqa"

batchsizes=( 8 12 16 20 )
for s in "${batchsizes[@]}"
do
    learningrates=( 2e-5 3e-5 5e-5 )

    for l in "${learningrates[@]}"
    do
        epochs=( 10 )

        for e in "${epochs[@]}"
        do
            python run_multiway_att.py --task_name "${task}" --do_eval --do_train --do_lower_case --bert_model bert-large-uncased --data_dir data/ --max_seq_length 220 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e}  --gradient_accumulation_steps=4  --output_dir output/batch_${s}_lr_${l}_epochs${e}_seed_42 --fp16 --seed 42
        done
    done
done
