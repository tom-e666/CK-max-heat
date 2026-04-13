python.exe train.py --device cuda --epochs 30 --batch-size 16 --num-workers 4 --lr 0.001
python.exe train.py --device cuda --epochs 30 --batch-size 128 --num-workers 4 --lr 0.001

python.exe train.py --device cuda --epochs 30 --batch-size 16 --num-workers 4 --lr 0.001 --embedding-dim 128 --proto-momentum 0.95 --boundary-margin 0.15 --push-scale 0.5 --lambda-ae 0.02 --lambda-push 0.25