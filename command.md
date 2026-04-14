python.exe train.py --device cuda --epochs 30 --batch-size 16 --num-workers 4 --lr 0.001
python.exe train.py --device cuda --epochs 30 --batch-size 128 --num-workers 4 --lr 0.001

python.exe train.py --device cuda --epochs 30 --batch-size 16 --num-workers 4 --lr 0.001 --embedding-dim 128 --proto-momentum 0.95 --boundary-margin 0.15 --push-scale 0.5 --lambda-ae 0.02 --lambda-push 0.25 --aux-start-epoch 4 --aux-ramp-epochs 6 --print-confusion --print-multilabel-confusion

python.exe train.py --device cuda --epochs 30 --batch-size 16 --num-workers 4 --lr 0.0005 --embedding-dim 128 --proto-momentum 0.95 --boundary-margin 0.10 --push-scale 0.4 --lambda-ae 0.01 --lambda-push 0.12 --aux-start-epoch 6 --aux-ramp-epochs 10 --print-confusion --print-multilabel-confusion

python.exe train.py --device cuda --epochs 40 --batch-size 16 --num-workers 4 --lr 0.0005 --weight-decay 0.0003 --embedding-dim 128 --dropout 0.35 --label-smoothing 0.05 --proto-momentum 0.95 --boundary-margin 0.10 --push-scale 0.4 --lambda-ae 0.01 --lambda-push 0.12 --aux-start-epoch 6 --aux-ramp-epochs 10 --lr-factor 0.5 --lr-patience 3 --patience 8 --min-delta 0.001 --print-confusion --print-multilabel-confusion