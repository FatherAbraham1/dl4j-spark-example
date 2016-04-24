package org.deeplearning4j.examples.cnn;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Simple example of learning MNIST with spark (local)
 * NOTE: This example runs and gives reasonable results, but better performance could be obtained
 * with some additional tuning of network hyperparameters
 *
 * @author Alex Black
 */
public class MnistExample {
    private static final Logger log = LoggerFactory.getLogger(MnistExample.class);

    public static void main(String[] args) throws Exception {

        //Create spark context
        int nCores = 1; //Number of CPU cores to use for training
        SparkConf sparkConf = new SparkConf();
//        sparkConf.setMaster("local[" + nCores + "]");
        sparkConf.setAppName("MNIST");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        int nChannels = 1;
        int outputNum = 10;
        int numSamples = 60000;
        int nTrain = 50000;
        int nTest = 10000;
        int batchSize = 60;
        int iterations = 1;
        int seed = 123;
        int nEpochs = 1;

        //Load data into memory
        log.info("Load data....");
        DataSetIterator mnistIter = new MnistDataSetIterator(1, numSamples, true);
        List<DataSet> allData = new ArrayList<>(numSamples);
        while (mnistIter.hasNext()) {
            allData.add(mnistIter.next());
        }
        Collections.shuffle(allData, new Random(12345));

        Iterator<DataSet> iter = allData.iterator();
        List<DataSet> train = new ArrayList<>(nTrain);
        List<DataSet> test = new ArrayList<>(nTest);

        int c = 0;
        while (iter.hasNext()) {
            if (c++ <= nTrain) train.add(iter.next());
            else test.add(iter.next());
        }

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());
        MultiLayerNetwork net;
        File f = new File("model/coefficients.bin");
        if (f.exists() && !f.isDirectory()) {
            log.info("load model...");
            //Load parameters from disk:
            INDArray newParams;
            try (DataInputStream dis = new DataInputStream(new FileInputStream("model/coefficients.bin"))) {
                newParams = Nd4j.read(dis);
            }

            //Load network configuration from disk:
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration
                    .fromJson(FileUtils.readFileToString(new File("model/conf.json")));

            //Create a MultiLayerNetwork from the saved configuration and parameters
            net = new MultiLayerNetwork(confFromJson);
            net.init();
            net.setParameters(newParams);

            //Load the updater:
            org.deeplearning4j.nn.api.Updater updater;
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("model/updater.bin"))) {
                updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
            }

            //Set the updater in the network
            net.setUpdater(updater);
        } else {
            //Set up network configuration
            log.info("Build model....");
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .regularization(true).l2(0.0005)
                    .learningRate(0.01)//.biasLearningRate(0.02)
                    //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list(6)
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(nChannels)
                            .stride(1, 1)
                            .nOut(20)
                            .activation("identity")
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .nIn(nChannels)
                            .stride(1, 1)
                            .nOut(50)
                            .activation("identity")
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(4, new DenseLayer.Builder().activation("relu")
                            .nOut(500).build())
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(outputNum)
                            .activation("softmax")
                            .build())
                    .backprop(true).pretrain(false);
            new ConvolutionLayerSetup(builder, 28, 28, 1);

            MultiLayerConfiguration conf = builder.build();
            net = new MultiLayerNetwork(conf);
            net.init();
            net.setUpdater(null);   //Workaround for minor bug in 0.4-rc3.8
        }

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net);

        //Train network
        log.info("--- Starting network training ---");
        for (int i = 0; i < nEpochs; i++) {
            //Run learning. Here, we are training with approximately 'batchSize' examples on each executor
            net = sparkNetwork.fitDataSet(sparkDataTrain, nCores * batchSize);
            System.out.println("----- Epoch " + i + " complete -----");

            log.info("Sve configure file to hdfs");
            //Write the network parameters:
            try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("model/coefficients.bin")))) {
                Nd4j.write(net.params(), dos);
            }

            //Evaluate (locally)
            Evaluation eval = new Evaluation();
            for (DataSet ds : test) {
                INDArray output = net.output(ds.getFeatureMatrix());
                eval.eval(ds.getLabels(), output);
            }
            log.warn(eval.stats());
            log.info("****************Example finished********************");

            //Write the network configuration:
            FileUtils.write(new File("model/conf.json"), net.getLayerWiseConfigurations().toJson());

            //Save the updater:
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model/updater.bin"))) {
                oos.writeObject(net.getUpdater());
            }
        }
    }
}
