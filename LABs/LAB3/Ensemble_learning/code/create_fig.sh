#!/bin/bash -e

# ln -sfn means create a symbolic link, forcefully replacing any existing link
# s means symbolic link
# f means force (remove existing destination files)
# n means no dereference (treat destination that is a symlink as a normal file)

get_newest_folder() {
    ls -td ../results/*/ | head -1 | xargs basename
}

run() {
    # print command with capitalized last argument as method name
    echo "$@" | awk '{print toupper(substr($NF, 1, 1)) substr($NF, 2) ":"}'
    # execute command and filter output for accuracy and F1 score
    "$@" | grep -oE "Acc=[0-9.]+, F1=[0-9.]+"
}

# conda activate ai25
export MPLBACKEND=AGG
export PYTHONWARNINGS=ignore::UserWarning

# Run training scripts

run python train.py --method voting
ln -sfn ../results/"$(get_newest_folder)" ../results/latest_voting
run python train.py --method bagging
ln -sfn ../results/"$(get_newest_folder)" ../results/latest_bagging
run python train.py --method adaboost
ln -sfn ../results/"$(get_newest_folder)" ../results/latest_adaboost
run python train.py --learning_rate 1 --method gbdt
ln -sfn ../results/"$(get_newest_folder)" ../results/latest_gbdt
run python train.py --method stacking
ln -sfn ../results/"$(get_newest_folder)" ../results/latest_stacking

# Copy confusion matrix images
cp ../results/latest_voting/voting_cm.png ../../assets/
cp ../results/latest_bagging/bagging_cm.png ../../assets/
cp ../results/latest_adaboost/adaboost_cm.png ../../assets/
cp ../results/latest_gbdt/gbdt_cm.png ../../assets/
cp ../results/latest_stacking/stacking_cm.png ../../assets/

# Copy training curves
cp ../results/latest_adaboost/adaboost_training_curve.png ../../assets/
# cp ../results/latest_gbdt/gbdt_training_curve.png ../../assets/