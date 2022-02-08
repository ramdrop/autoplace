# with DTR
DATASET_DIR='./../dataset'
STR_DIR="${DATASET_DIR}/7n5s_xy11_removal"
echo $DATASET_DIR
echo $STR_DIR

python config.py --remove --dataset_dir=$DATASET_DIR
python split.py --struct_dir=$STR_DIR
python split_refine.py --struct_dir=$STR_DIR
python convert.py --split='test' --struct_dir=$STR_DIR --img=1 --pcl=1 --rcs=1
python convert.py --split='trainval' --struct_dir=$STR_DIR --img=1 --pcl=1 --rcs=1

