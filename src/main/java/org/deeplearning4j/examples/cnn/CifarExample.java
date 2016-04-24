package org.deeplearning4j.examples.cnn;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
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
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
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
public class CifarExample {

    private static final int WIDTH = 32;
    private static final int HEIGHT = 32;
    private static final int CHANNELS = 3;
    private static final int BATCH_SIZE = 60;
    private static final int ITERATIONS = 1;
    private static final int SEED = 123;
    private static final int SAMPLE_NUM = 50000;
    private static final int TRAIN_NUM = 42000;
    private static final int TEST_NUM = 8000;
    private static final Logger log = LoggerFactory.getLogger(CifarExample.class);

    public static void main(String[] args) throws Exception {

        //Create spark context
        int nCores = 1; //Number of CPU cores to use for training
        int nEpochs = 1;
        SparkConf sparkConf = new SparkConf();
//        sparkConf.setMaster("local[" + nCores + "]");
        sparkConf.setAppName("CIFAR");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load data into memory
        log.info("****************Load data****************");

        String labeledPath = System.getProperty("user.home") + "/downloads/cifar";
        List<String> labels = new ArrayList<>();
        for (File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }
        System.out.println(labels);


        RecordReader recordReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, true, labels);

        recordReader.initialize(new FileSplit(new File(labeledPath)));

        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,
                1, WIDTH * HEIGHT * CHANNELS, labels.size());

        List<DataSet> allData = new ArrayList<>();
        while (iter.hasNext()) {
            allData.add(iter.next());
        }

        Collections.shuffle(allData, new Random(12345));

        Iterator<DataSet> iters = allData.iterator();
        List<DataSet> train = new ArrayList<>(TRAIN_NUM);
        List<DataSet> test = new ArrayList<>(TEST_NUM);

        int c = 0;
        while (iters.hasNext()) {
            if (c++ <= TRAIN_NUM) train.add(iters.next());
            else test.add(iters.next());
        }

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
//        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());File f = new File("model/coefficients.bin");
        MultiLayerNetwork net;
        File f = new File("model/c_coefficients.bin");
        if (f.exists() && !f.isDirectory()) {
            log.info("load model...");
            //Load parameters from disk:
            INDArray newParams;
            try (DataInputStream dis = new DataInputStream(new FileInputStream("model/c_coefficients.bin"))) {
                newParams = Nd4j.read(dis);
            }

            //Load network configuration from disk:
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration
                    .fromJson(FileUtils.readFileToString(new File("model/c_conf.json")));

            //Create a MultiLayerNetwork from the saved configuration and parameters
            net = new MultiLayerNetwork(confFromJson);
            net.init();
            net.setParameters(newParams);

            //Load the updater:
            org.deeplearning4j.nn.api.Updater updater;
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("model/c_updater.bin"))) {
                updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
            }

            //Set the updater in the network
            net.setUpdater(updater);
        } else {

            //Set up network configuration
            log.info("Build model....");
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(SEED)
                    .iterations(ITERATIONS)
                    .momentum(0.9)
                    .regularization(true)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.ADAGRAD)
                    .list(6)
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(CHANNELS)
                            .stride(1, 1)
                            .nOut(20)
                            .weightInit(WeightInit.XAVIER)
                            .activation("relu")
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .nIn(20)
                            .nOut(40)
                            .stride(1, 1)
                            .weightInit(WeightInit.XAVIER)
                            .activation("relu")
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .build())
                    .layer(4, new DenseLayer.Builder()
                            .activation("relu")
                            .weightInit(WeightInit.XAVIER)
                            .nOut(1000)
                            .dropOut(0.5)
                            .build())
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(labels.size())
                            .dropOut(0.5)
                            .weightInit(WeightInit.XAVIER)
                            .activation("softmax")
                            .build())
                    .inputPreProcessor(4, new CnnToFeedForwardPreProcessor())
                    .backprop(true)
                    .pretrain(false);
            new ConvolutionLayerSetup(builder, WIDTH, HEIGHT, CHANNELS);

            MultiLayerConfiguration conf = builder.build();
            net = new MultiLayerNetwork(conf);
            net.init();
            net.setUpdater(null);
        }

        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net);


        log.info("****************Starting network training****************");
        //Run learning. Here, we are training with approximately 'batchSize' examples on each executor
        for (int i = 0; i < nEpochs; i++) {
            log.info("Epoch " + i + "Start");
            net = sparkNetwork.fitDataSet(sparkDataTrain, nCores * BATCH_SIZE);
            log.info("Epoch " + i + "Complete");

            log.info("****************Starting Evaluation********************");
            Evaluation eval = new Evaluation();
            for (DataSet ds : test) {
                INDArray output = net.output(ds.getFeatureMatrix());
                eval.eval(ds.getLabels(), output);
            }
            log.warn(eval.stats());

            log.info("****************Save configure files****************");
            try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("model/c_coefficients.bin")))) {
                Nd4j.write(net.params(), dos);
            }
            FileUtils.write(new File("model/c_conf.json"), net.getLayerWiseConfigurations().toJson());
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model/c_updater.bin"))) {
                oos.writeObject(net.getUpdater());
            }
        }


    }
}
