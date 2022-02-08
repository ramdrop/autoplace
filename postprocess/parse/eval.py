# --------------------------------------------- Eval --------------------------------------------- #
import os
import json

with open('postprocess/parse/resume_path.json', 'r') as f:
    record = json.load(f)

for dut in ['encoder', 'encoder_dtr', 'encoder_lstm', 'encoder_dtr_lstm']:
    cmds = "python train.py --mode='evaluate'  --cGPU=0 --split=test --resume={}".format(record[dut])
    exc_state = os.system(cmds)
    assert exc_state == 0, 'unexpected failure'

# for dut in ['s3_seqnet', 's1_seqnet']:
#     cmds = "python train_seqnet.py --mode=evaluate  --split=test --cGPU=1 --resume={}".format(record[dut])
#     exc_state = os.system(cmds)
#     assert exc_state == 0, 'unexpected failure'

# cmds = "python train_kid.py --mode=evaluate  --split=test --cGPU=1 --resume={}".format(record['kid'])
# exc_state = os.system(cmds)
# assert exc_state == 0, 'unexpected failure'

# for dut in ['netvlad', 'netvlad_lstm']:
#     cmds = "python train.py --mode=evaluate  --split=test --cGPU=1 --resume={}".format(record[dut])
#     exc_state = os.system(cmds)
#     assert exc_state == 0, 'unexpected failure'
