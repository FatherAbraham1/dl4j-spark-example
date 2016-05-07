#!/usr/bin/env bash
spark-submit --class org.deeplearning4j.cnn.CifarEpoch --master spark://student32-x1:7077 --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./src/main/resources/log4j.properties" ./target/dl4j-spark-cdh5-examples-1.0-SNAPSHOT.jar
