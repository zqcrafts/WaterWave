NAME=uw3_mini
ROOT_DIR=/home/ubuntu/data/code/waterwave_code/flow_estimate/all_sequences
CODE_DIR=/home/ubuntu/data/code/waterwave_code/flow_estimate/data_preprocessing/RAFT

IMG_DIR=$ROOT_DIR/${NAME}
FLOW_DIR=$ROOT_DIR/${NAME}_flow
CONF_DIR=${FLOW_DIR}_confidence

CUDA_VISIBLE_DEVICES=0 \
python ${CODE_DIR}/demo.py \
--model=${CODE_DIR}/models/raft-sintel.pth \
--path=$IMG_DIR \
--outdir=$FLOW_DIR \
--name=$NAME \
--confidence \
--outdir_conf=$CONF_DIR
