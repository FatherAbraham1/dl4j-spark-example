package org.deeplearning4j.examples.cnn;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Vannahz
 */
public class TestModel {
	
	private static final Logger log = LoggerFactory.getLogger(TestModel.class);
	
	public static void main(String[] args) throws Exception{
		
		//Create spark context
        int nCores = 6;
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nCores + "]");
        sparkConf.setAppName("MNIST");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        
        //------Simple Dataset--------
        List<DataSet> test = new ArrayList<>(1);
		BufferedReader bf;
		float[] input = new float[784];
		bf = new BufferedReader(new FileReader( new File("input.txt") ));
		String[] s = bf.readLine().split(",");
		for(int i=0; i<s.length; i++) input[i] = Float.valueOf(s[i]);
		
		INDArray array1 = Nd4j.create(input);
		INDArray array2 = Nd4j.create(new int[]{1});
		DataSet inds = new DataSet(array1, array2);
		test.add(inds);
        //----------------------------
		
        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(test);
        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());
        
		
		//Load network configuration from disk:
	    MultiLayerConfiguration confFromJson = 
	    		MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));

	    //Load parameters from disk:
	    INDArray newParams;
	    try(DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"))){
	        newParams = Nd4j.read(dis);
	    }
	    
	    //Load the updater:
	    org.deeplearning4j.nn.api.Updater updater;
	    try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream("updater.bin"))){
	        updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
	    }

	    //Create a MultiLayerNetwork from the saved configuration and parameters
	    MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
	    savedNetwork.init();
	    savedNetwork.setParameters(newParams);
	    savedNetwork.setUpdater(updater);
	    
	    SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, savedNetwork);
	    
	    //log
	    //Run learning. Here, we are training with approximately 'batchSize' examples on each executor
	    savedNetwork = sparkNetwork.fitDataSet(sparkDataTrain, 1);
        System.out.println("----- Conclusion -----");

        //Evaluate (locally)  改！
        Evaluation eval = new Evaluation();
        for(DataSet ds : test){
            INDArray output = savedNetwork.output(ds.getFeatureMatrix());
            System.out.println("**" + output);
            
            eval.eval(ds.getLabels(),output);
        }
        log.info(eval.stats());
	}
}
