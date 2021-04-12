python inference.py --weight './pre_trained/deform.pt' --src './samples/1/source.obj' --tgt './samples/1/target.obj' --if_nonrigid 1 --iteration 7 --device_id 0

python inference.py --weight './pre_trained/deform.pt' --src './samples/2/source.obj' --tgt './samples/2/target.obj' --if_nonrigid 1 --iteration 7 --device_id 0

python inference.py --weight './pre_trained/modelnet40.pt' --src './samples/3/source.obj' --tgt './samples/3/target.obj' --if_nonrigid 0 --iteration 11 --device_id 0

python inference.py --weight './pre_trained/modelnet40.pt' --src './samples/4/source.obj' --tgt './samples/4/target.obj' --if_nonrigid 0 --iteration 11 --device_id 0
