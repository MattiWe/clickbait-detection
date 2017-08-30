#! /bin/bash

out_directory="out"
input_directory="input.jsonl"
mapper="hadoop/hadoop_mapper.py"
reducer="hadoop/hadoop_reducer.py"

# 1. copy files to hadoop fs
# hadoop fs -put $mapper $mapper
# hadoop fs -put "hadoop/input" $input_directory

# 2. run mapreduce
hadoop jar "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar" \
       -D mapred.reduce.tasks=500 \
       -D stream.map.output.field.separator=? \
       -D yarn.app.mapreduce.am.resource.cpu-vcores=8 \
       -D mapreduce.reduce.cpu.vcores=2 \
       -D mapreduce.reduce.memory.mb=8192	\
       -file $mapper -mapper $mapper \
       -file $reducer -reducer $reducer \
       -file "hadoop/x_test.npz" \
       -file "hadoop/y_test.npz" \
       -file "hadoop/x_train.npz" \
       -file "hadoop/y_train.npz" \
       -input $input_directory \
       -output $out_directory

       # -D mapred.task.timeout=1800000 \
       # -files "hadoop" \
# 3. fetch results and update input files
# hadoop fs -cat $out_directory/* # TODO
