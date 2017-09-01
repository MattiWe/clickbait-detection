#! /bin/bash

out_directory="out"
input_directory="input.jsonl"
mapper="hadoop/hadoop_mapper.py"
reducer="hadoop/hadoop_reducer.py"

# 2. run mapreduce
hadoop jar "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar" \
       -D mapred.reduce.tasks=1000 \
       -file $mapper -mapper $mapper \
       -file $reducer -reducer $reducer \
       -file "hadoop/x_test.npz" \
       -file "hadoop/y_test.npz" \
       -file "hadoop/x_train.npz" \
       -file "hadoop/y_train.npz" \
       -input $input_directory \
       -output $out_directory
