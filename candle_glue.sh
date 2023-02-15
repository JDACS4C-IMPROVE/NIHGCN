#!/bin/bash

set -a
export PYTHONPATH=/usr/local/NIHGCN
export CANDLE_DATA_DIR=/candle_data_dir
set +a

mkdir -p $CANDLE_DATA_DIR/NIHGCN/Data

python get_test_data.py

#move directories around - use GDSC as test case
mv $CANDLE_DATA_DIR/common/Data/GDSC/*.csv $CANDLE_DATA_DIR/NIHGCN/Data/.
