#!/bin/bash

set -a
export PYTHONPATH=$IMPROVE_MODEL_DIR:$PYTHONPATH
set +a

if [ -z "$CANDLE_DATA_DIR" ]; then
  echo "CANDLE_DATA_DIR not set"
  exit 421
fi

mkdir -p $CANDLE_DATA_DIR/NIHGCN/Data

python get_test_data.py

#move directories around - use GDSC as test case
mv $CANDLE_DATA_DIR/common/Data/GDSC/*.csv $CANDLE_DATA_DIR/NIHGCN/Data/.
