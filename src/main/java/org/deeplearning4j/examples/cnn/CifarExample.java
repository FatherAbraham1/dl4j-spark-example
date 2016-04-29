package org.deeplearning4j.examples.cnn;

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

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * @author Vannahz
 */
public class CifarExample {
	
    private static final int WIDTH = 32;
    private static final int HEIGHT = 32;
    private static final int CHANNELS = 3;
    private static final int BATCH_SIZE = 20;//80;
    private static final int ITERATIONS = 1;
    private static final int SEED = 123;
    private static final int TRAIN_NUM = 21087;//25000;
    private static final int TEST_NUM = 1400;//5000;    //SAMPLE_NUM = 5000
    private static final Logger log = LoggerFactory.getLogger(CifarExample.class);

    public static void main(String[] args) throws Exception {

        //Create spark context
        int nCores = 4; //Number of CPU cores to use for training
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nCores + "]");
        sparkConf.setAppName("CIFAR");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load data into memory
        log.info("****************Load Train data****************");
        
	    String trainPath = System.getProperty("user.home")+ "/train";  //"/cifar/mitrain";
	    List<String> labels = new ArrayList<>(); 
	    for(File f : new File(trainPath).listFiles()) { 
	        labels.add(f.getName());
	    }
	    System.out.println(labels);
	    
	    
	    RecordReader recordReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, true, labels);
	    
	    recordReader.initialize(new FileSplit(new File(trainPath)));
	    
	    DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader,
	    								1, WIDTH * HEIGHT * CHANNELS, labels.size());
        
        List<DataSet> train = new ArrayList<>();
        while(trainIter.hasNext()){
            train.add(trainIter.next());
        }
        Collections.shuffle(train,new Random(12345));
        
        //----------------------------------------------------------------------------
        
        log.info("****************Load Test data****************");
        
	    String testPath = System.getProperty("user.home")+ "/test";  //"/cifar/mitrain";
	     
	    RecordReader testReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, true, labels);
	    
	    testReader.initialize(new FileSplit(new File(testPath)));
	    
	    DataSetIterator tsetIter = new RecordReaderDataSetIterator(testReader,
	    								1, WIDTH * HEIGHT * CHANNELS, labels.size());
        
        List<DataSet> test = new ArrayList<>();
        while(tsetIter.hasNext()){
            test.add(tsetIter.next());
        }
        
        Collections.shuffle(test,new Random(12543));
        //-----------------------------------------------------------------------------------------

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());

        //Set up network configuration
        log.info("Build model....");
        
        
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
        		.seed(SEED)
                .iterations(ITERATIONS)
                .regularization(true).l2(0.04)
                .learningRate(0.05)
                .weightInit(WeightInit.XAVIER) // consider standard distribution with std .05
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAGRAD)
                .momentum(0.9)
                .list(8)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                		.build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(32)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3})
                		.build())
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .nIn(32)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(64)
                        .activation("relu")
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3})
                		.build())
                .layer(6, new DenseLayer.Builder()
                		.activation("relu")
                        .nOut(64)
                        .dropOut(0.5)
                        .build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(labels.size())
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false);
        new ConvolutionLayerSetup(builder, WIDTH, HEIGHT, CHANNELS);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setUpdater(null);
        
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net);

        
        log.info("****************Starting network training****************");
        //Run learning. Here, we are training with approximately 'batchSize' examples on each executor
        net = sparkNetwork.fitDataSet(sparkDataTrain, nCores * BATCH_SIZE);
        //net = sparkNetwork.fitDataSet(sparkDataTrain);

        
        log.info("****************Starting Evaluation********************");
        Evaluation eval = new Evaluation();
        for(DataSet ds : test){
            INDArray output = net.output(ds.getFeatureMatrix());
            System.out.println(output);
            eval.eval(ds.getLabels(),output);
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
