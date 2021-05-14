run() {
  for X_DIM in 2 3 5; do
    for N in 200 300 500 1000; do
        sbatch single.sh $1 " \
          with \
            base.num_experiments=100000 \
            base.seed=0 \
            base.X_DIM=$X_DIM \
            base.N=$N \
            $2 \
        "
    done
  done
}

EQ=$1

if [ "$EQ" == "unstratified_distributed" ] || [ "$EQ" == "stratified_distributed" ] ; then
  BATCH_ARG="base.batch_size=32"
else
  BATCH_ARG="base.batch_size=128"
fi

if [ "$EQ" == "unstratified_pooled" ]; then
  run $EQ
else
  for K in 3 4 5; do
    run $EQ "grouped.K=${K} ${BATCH_ARG}"
  done
fi

