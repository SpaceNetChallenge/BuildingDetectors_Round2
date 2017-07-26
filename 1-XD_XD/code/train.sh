#!/bin/bash
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

echo ">>> CLEAN UP"
echo rm -rf /data/working
rm -rf /data/working && mkdir -p /data/working

source activate py35 && for train_path in $@; do
    echo ">>> PREPROCESSING STEP"
    echo python v5_im.py preproc_train $train_path
    python v5_im.py preproc_train $train_path
    echo python v12_im.py preproc_train $train_path
    python v12_im.py preproc_train $train_path
    echo python v16.py preproc_train $train_path
    python v16.py preproc_train $train_path

    echo ">>> TRAINING v9s model"
    echo python v9s.py validate $train_path
    python v9s.py validate $train_path
    echo python v9s.py evalfscore $train_path
    python v9s.py evalfscore $train_path

    ### v13 --------------
    # Training for v13 model
    echo ">>>>>>>>>> v13.py"
    python v13.py validate $train_path
    # Parametr optimization for v13 model
    echo ">>>>>>>>>> v13.py"
    python v13.py evalfscore $train_path

    ### v16 --------------
    # Training for v16 model
    echo ">>>>>>>>>> v16.py"
    python v16.py validate $train_path
    # Parametr optimization for v16 model
    echo ">>>>>>>>>> v16.py"
    python v16.py evalfscore $train_path

    ### v17 --------------
    echo ">>>>>>>>>> v17.py"
    python v17.py evalfscore $train_path
done
