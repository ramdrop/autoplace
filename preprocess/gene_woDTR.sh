# without DTR
DATASET_DIR='./../dataset'
ST_DIR="${DATASET_DIR}/7n5s_xy11"
echo $DATASET_DIR
echo $ST_DIR

python config.py --dataset_dir=$DATASET_DIR
python split.py --struct_dir=$ST_DIR
python split_refine.py --struct_dir=$ST_DIR
python convert.py --split='test' --struct_dir=$ST_DIR --img=1 --pcl=1 --rcs=1
python convert.py --split='trainval' --struct_dir=$ST_DIR --img=1 --pcl=1 --rcs=1

