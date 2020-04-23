import logging
import os

import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
parser = argparse.ArgumentParser(description='CoTraining')
parser.add_argument('--top_k', type=int,
                    help='Number of labeling samples in Co-Training')
parser.add_argument('--gpu', type=int,
                    help='gpu id')
args = parser.parse_args()

top_k = args.top_k
recurrent_times = 20

load_evidence_file = 'none'

for i in range(recurrent_times):
    logger.info("Running at the %dth times..." % (i))
    save_evidence_file = "recurrent%d" % (i)
    output_dir = "models/selftraining-SingleEvidence/top%d/recurrent%d" % (top_k, i)
    os.makedirs(output_dir)

    if i > 0:
        previous_output_dir = "models/selftraining-SingleEvidence/top%d/recurrent%d" % (top_k, i - 1)
        os.system("cp %s/quasart_all.%s.pkl %s/quasart_all.%s.pkl" % (previous_output_dir, load_evidence_file, output_dir, load_evidence_file))

    cmd = "env CUDA_VISIBLE_DEVICES=%d \
            python main.py \
            --gpu %d \
            --batch-size 32 \
            --model-name quasart_all \
            --num-epochs 5 \
            --dataset quasart \
            --mode all \
            --pretrained models/quasart_selector.mdl \
            --model-dir %s \
            --top_k %d \
            --load_evidence_file %s \
            --save_evidence_file %s" % (args.gpu, 0, output_dir, top_k, load_evidence_file, save_evidence_file)

    os.system(cmd)

    load_evidence_file = save_evidence_file
