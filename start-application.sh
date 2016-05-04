#!/usr/bin/env bash
spark-submit --class org.deeplearning4j.examples.cnn.CifarEpoch --master yarn-client ./target/dl4j-spark-cdh5-examples-1.0-SNAPSHOT.jar
