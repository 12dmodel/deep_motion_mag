# must set CUDA_VISIBLE_DEVICES on the shell, i.e. calling
#   CUDA_VISIBLE_DEVICES=0 sh scripts/run_exp.sh urban_cg2real_ac
EXP_NAME="$1"
VIDEO="$2"
AMPLIFICATION_FACTOR="$3"
DYNAMIC_MODE=${4:-"no"}
OUT_DIR=data/output/"$VIDEO"_"$EXP_NAME"_"$AMPLIFICATION_FACTOR"
VID_DIR=data/vids/"$VIDEO"
OUT_DIR=data/output/"$VIDEO"
if [ ! -d "$OUT_DIR" ]; then
    mkdir -p "$OUT_DIR"
fi

FLAGS="--phase=run --vid_dir=$VID_DIR --out_dir=$OUT_DIR --amplification_factor=$AMPLIFICATION_FACTOR"
if [ "$DYNAMIC_MODE" = yes ] ; then
    FLAGS="$FLAGS"" --velocity_mag"
fi
python main.py --config_file=configs/"$EXP_NAME".conf \
    $FLAGS

