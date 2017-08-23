#! /bin/bash

out_directory="out"
input_directory="input"
mapper="hadoop_mapper.py"
reducer="hadoop_reducer.py"

# 1. copy files to hadoop fs
hadoop fs put $mapper $mapper
hadoop fs put $reducer $reducer
# TODO put npzs on hdfs
# put module on hdfs

# 2. run mapreduce # TODO append ml module
hadoop jar "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar" \
       -file $mapper -mapper $mapper \
       -file $reducer -reducer $reducer \
       -input $input_directory \
       -output $out_directory \ # numebred
       -D mapred.reduce.tasks=100

# 3. fetch results and update input files
hadoop fs -cat $out_directory/* # TODO
