
### Train/Evaluate SOTA methods

1. NetVLAD/NetVLAD+LSTM

    We develop the testbench based on [Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad).

    train NetVLAD/NetVLAD+LSTM
    ```bash
    # NetVLAD
    cd autoplace
    python train.py  --nEpochs=40 --output_dim=9216 --seqLen=1 --encoder_dim=512 --num_clusters=18 --net=netvlad --split=val --logsPath=logs_netvlad --cGPU=0 --imgDir='dataset/7n5s_xy11/img'

    # NetVLAD+LSTM
    cd autoplace
    python train.py  --nEpochs=40 --output_dim=4096 --seqLen=3 --encoder_dim=512 --num_clusters=18 --net=netvlad --split=val --logsPath=logs_netvlad --cGPU=0 --imgDir='dataset/7n5s_xy11/img'    
    ```
    evaluate NetVLAD/NetVLAD+LSTM
    ```bash
    cd autoplace
    python train.py --mode='evaluate'  --cGPU=0  --split=test --resume=[log_folder]
    ```
    calculate Recall@N and Precision-Recall: modify the path `netvlad` in `autoplace/postprocess/parse/resume_path.json` to [netvlad_log_folder], and run the command:
    ```bash
    cd autoplace/postprocess/parse
    python parse.py --model=netvlad
    ```


2. SeqNet

    We develop the testbench based on [oravus/SeqNet](https://github.com/oravus/seqNet). We use the pretrained NetVLAD as the black-box feature encoder to encode images to 9216D vectors, which is then downscaled to 4096D using PCA: modify the path `netvlad` in `postprocess/parse/resume_path.json` to [netvlad_log_folder] and run the command:
    ```bash
    cd autoplace/postprocess/parse
    python parse_seqnet.py --phase='generate_features'
    ```

    use the stored 4096D vectors to train S3-SeqNet and S1-SeqNet (it can be done in parallel):
    ```bash
    # train S3-SeqNet
    cd autoplace
    python train_seqnet.py  --split=val --cGPU=0 
  
    # train S1-SeqNet
    cd autoplace
    python train_seqnet.py  --split=val --seqLen=1 --w=1 --cGPU=0 
    ```
    evaluate
    ```bash
    # evaluate S3-SeqNet or S1-SeqNet
    cd autoplace
    python train_seqnet.py --mode='evaluate'  --cGPU=0  --split=test --resume=[S3-SeqNet or S1-SeqNet training log folder]
    ```
    generate detailed results: modify the path `s3_seqnet` and `s1_seqnet` in `postprocess/parse_logs/resume_path.json` to their [log_folder] and run the command:
    ```bash
    cd autoplace/postprocess/parse
    python parse_seqnet.py --phase='match'
    ```


3. MinkLoc3D

    refer to [minkloc3d_AutoPlace](https://github.com/ramdrop/minkloc3d_AutoPlace) to generate `minkloc3d_features.pickle`

    then copy `minkloc3d_features.pickle` to `autoplace/postprocess/parse/results`
    
    run the command:
    ```bash
    cd autoplace/postprocess/parse
    python parse.py --model='minkloc3d'
    ```



4. ScanContext

    We develop the testbench based on [irapkaist/scancontext](https://github.com/irapkaist/scancontext).

    ```bash
    cd autoplace/postprocess/parse
    python parse.py --model='scancontext'
    ```



5. M2DP

    We develop the testbench based on [M2DP python](https://pypi.org/project/m2dp/).

    ```bash
    cd autoplace/postprocess
    python parse.py --model='m2dp'
    ```

6. Under The Radar

    refer to [utr_AutoPlace](https://github.com/ramdrop/utr_AutoPlace) repo to generate `utr_features.pickle`.

    copy `utr_features.pickle` in `autoplace/postprocess/parse/results`
    
    run the command:
    ```bash
    cd autoplace/postprocess/parse
    python parse.py --model='utr'
    ```    

7. Kidnapped Radar

    convert Cartesian images to Polar images
    ```bash
    cd autoplace/preprocess
    python cart_to_polar.py --dataset='../dataset/7n5s_xy11'
    ```

    train
    ```bash
    cd autoplace
    python train_kid.py  --nEpochs=80 --output_dim=32768 --seqLen=1 --encoder_dim=512 --num_clusters=64 --net=kid --logsPath=logs_kid --split=val --cGPU=0 --imgDir='dataset/7n5s_xy11/img_polar'
    ```
    evaluate
    ```bash
    cd autoplace
    python train_kid.py --mode='evaluate'  --cGPU=0  --split=test --resume=[training log folder]
    ```
    generate detailed results
    ```bash
    cd autoplace/postprocess/parse
    python parse.py --model='kidnapped'
    ```


### Visualize Recall and Precision-Recall
After training & evaluating the above competing methods, `[methods_name]_result.pickle` files should be in the `autoplace/postprocess/parse/results` folder. 

To generate (1) Reall@N curve (2) PR curve, (3) F1 Score, (4) Average Pecision, run
```bash
cd autoplace/postprocess/vis
python competing_figure.py 
python competing_scores.py 
```
