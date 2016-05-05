#!/usr/bin/env bash
spark-submit --class org.deeplearning4j.cnn.CifarEpoch --master local[4] --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./src/main/resources/log4j.properties" ./target/dl4j-spark-cdh5-examples-1.0-SNAPSHOT.jar
