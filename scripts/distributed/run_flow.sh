eq=$1
simulated_data_dir=$2
analysis_dir=$simulated_data_dir/$3

CMD="varderiv.distributed.cmd ${eq}"

local_send_T() {
    echo "Local send T..."
    for local_data in $(ls $analysis_dir | egrep -i '^local_[0-9]+.npz' ); do
        local_data="${analysis_dir}/${local_data}"
        python -m $CMD local-send-T $local_data \
            --save-dir=$analysis_dir;
    done
}

master_send_T() {
    echo "Master send T..."
    python -m $CMD master-send-T $analysis_dir \
        --save-dir=$analysis_dir
}

local_send_vals() {
    echo "Local send vals..."
    local T_delta
    if [ $eq == "unstratified_distributed" ]; then
        T_delta="${analysis_dir}/T_delta.npz"
    else
        T_delta=""
    fi
    for local_data in $(ls $analysis_dir | egrep -i '^local_[0-9]+.npz' ); do
        local_data="${analysis_dir}/${local_data}"
        python -m $CMD local-send-vals $T_delta $local_data \
           --save-dir=$analysis_dir \
           --solver-thres=1e-3 --solver-max-steps=300;
    done
}

master_analytics() {
    echo "Master analytics..."
    local analytics_arg
    case $eq in
        unstratified_pooled|stratified_pooled)
            analytics_arg="$analysis_dir/all.npz"
            ;;
        unstratified_distributed|stratified_distributed)
            analytics_arg=$analysis_dir
            ;;
    esac
    python -m $CMD master-analytics $analytics_arg \
       --save-dir=$analysis_dir \
       --solver-thres=1e-3 --solver-max-steps=300;
}

run_flow() {
    case $eq in
        unstratified_pooled)
            master_analytics
            ;;
        unstratified_distributed)
            local_send_T
            master_send_T
            local_send_vals
            master_analytics
            ;;
        stratified_pooled)
            master_analytics
            ;;
        stratified_distributed)
            local_send_vals
            master_analytics
            ;;
    esac
}

run_flow
