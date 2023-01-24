
# ROOT="/local-scratch/nishant/simul/mma"
ROOT="path/to/working/dir"

DATA="${ROOT}/data/data-bin/iwslt14.tokenized.de-en"

EXPT="${ROOT}/experiments/trial"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/fairseq"

USR="${ROOT}/simultaneous_translation"




export CUDA_VISIBLE_DEVICES=0

# infinite lookback

mma_il(){
    CKPT="${EXPT}/checkpoints/infinite"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation \
    --simul-type infinite_lookback \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --latency-weight-avg  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 7000 \
    --ddp-backend no_c10d
}

# hard align
mma_h(){
    CKPT="${EXPT}/checkpoints/hard"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation \
    --simul-type hard_aligned \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --latency-weight-avg  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 7000 \
    --ddp-backend no_c10d
}


# monotnic wait k
mma_wait_k(){
    CKPT="${EXPT}/checkpoints/waitk"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation \
    --simul-type waitk \
    --waitk-lagging 3 \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --latency-weight-avg  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 7000 \
    --ddp-backend no_c10d
}

# mma_il
# mma_h
mma_wait_k