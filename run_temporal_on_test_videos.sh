# must set CUDA_VISIBLE_DEVICES on the shell, i.e. calling
#   CUDA_VISIBLE_DEVICES=0 sh scripts/run_exp.sh urban_cg2real_ac
EXP_NAME="$1"
VIDEO="$2"
AMPLIFICATION_FACTOR="$3"
FL="$4"
FH="$5"
FS="$6"
N_TAP="$7"
FILTER_TYPE="$8"
VID_DIR=data/vids/"$VIDEO"
OUT_DIR=data/output/"$VIDEO"
python main.py --config_file=configs/"$EXP_NAME".conf \
    --phase=run_temporal \
    --vid_dir="$VID_DIR" \
    --out_dir="$OUT_DIR" \
    --amplification_factor="$AMPLIFICATION_FACTOR" \
    --fl="$FL" \
    --fh="$FH" \
    --fs="$FS" \
    --n_filter_tap="$N_TAP" \
    --filter_type="$FILTER_TYPE"

