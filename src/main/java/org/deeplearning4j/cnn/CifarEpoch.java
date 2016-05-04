package org.deeplearning4j.cnn;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
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
 * @author Vannahz
 */
public class CifarEpoch {

    private static final int WIDTH = 32;
    private static final int HEIGHT = 32;
    private static final int CHANNELS = 3;
    private static final int BATCH_SIZE = 6;
    private static final int ITERATIONS = 1;
    private static final int SEED = 123;
    private static final List<String> LABELS = Arrays.asList("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck");
    private static final Logger log = LoggerFactory.getLogger(CifarEpoch.class);

    public static void main(String[] args) throws Exception {

        
        int nCores = 10;
        int nEpochs = 1;
        SparkConf sparkConf = new SparkConf();
//        sparkConf.setMaster("local[" + nCores + "]");
        sparkConf.setAppName("CIFAR");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrationRequired", "true");
        sparkConf.set("spark.default.parallelism", "" + nCores * 3);
        sparkConf.set("spark.kryo.registrator", "util.HydraKryoSerializer");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load data into memory
        log.info("****************Load data****************");

        String labeledPath = System.getProperty("user.home") + "/cifar/train";
        RecordReader recordReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, true, LABELS);
        recordReader.initialize(new FileSplit(new File(labeledPath)));
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,
                1, WIDTH * HEIGHT * CHANNELS, LABELS.size());

        List<DataSet> train = new ArrayList<>();
        while (iter.hasNext()) {
            train.add(iter.next());
        }
        Collections.shuffle(train, new Random(12345));

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
        sparkDataTrain.persist(StorageLevel.MEMORY_AND_DISK());
        
        MultiLayerNetwork net;
        File f = new File("model/c_coefficients.bin");
        if (f.exists() && !f.isDirectory()) {
        	
            log.info("load model...");
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
                    .miniBatch(true)
                    .iterations(ITERATIONS)
                    .regularization(true).l1(1e-1).l2(2e-4)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.ADAGRAD)
                    .list(6)
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(CHANNELS)
                            .nOut(5)
                            .dropOut(0.5)
                            .weightInit(WeightInit.XAVIER)
                            .activation("relu")
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(3, 3)
                            .nOut(10)
                            .dropOut(0.5)
                            .weightInit(WeightInit.UNIFORM)
                            .activation("relu")
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2, 2})
                            .build())
                    .layer(4, new DenseLayer.Builder()
                            .activation("relu")
                            .nOut(100)
                            .build())
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(LABELS.size())
                            .weightInit(WeightInit.UNIFORM)
                            .activation("softmax")
                            .build())
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


            log.info("****************Starting Evaluation********************");
            Evaluation eval = new Evaluation();
            for (DataSet ds : train) {

                INDArray output = net.output(ds.getFeatureMatrix());
                //  if(j > 25000)
//                	System.out.println(output);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());


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
