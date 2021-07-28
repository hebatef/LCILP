# ClustLP - Clustering for Link Prediction

To start training a GraIL model, run the following command. python train.py -d WN18RR_v1 -e grail_wn_v1

To test GraIL run the following commands.

python test_auc.py -d WN18RR_v1_ind -e grail_wn_v1
python test_ranking.py -d WN18RR_v1_ind -e grail_wn_v1
