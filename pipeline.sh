python main.py --gpu=0 --batch-size 256 --model-name quasart_reader --num-epochs 10 --dataset quasart --mode reader
python main.py --gpu=0 --batch-size 64 --model-name quasart_selector --num-epochs 10 --dataset quasart --mode selector --pretrained models/quasart_reader.mdl