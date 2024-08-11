train_hash=$1

echo "Model: $1\n"

python leaderboard_eval.py \
  -lp "/home/Data/leaderboard" \
  -yp "reconstructions/${train_hash}"