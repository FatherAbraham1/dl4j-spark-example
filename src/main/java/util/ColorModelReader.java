package org.deeplearning4j.examples.cnn;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * @author Vannahz
 */
public class ColorModelReader {

    private static final Logger log = LoggerFactory.getLogger(ColorModelReader.class);

    public static void main(String[] args) throws Exception {
    	
    	String[] testlabel = {"airplane", "automobile", "bird", "cat", "deer",
    			"dog", "frog", "horse", "ship", "truck"};
    	
    	int model = colorModelReader("D:/test");
    	
    	System.out.println("This is " + testlabel[model]);
    	
    }
    
    static int colorModelReader(String filepath)throws Exception{
    	
    	
	    List<String> labels = new ArrayList<>(); 
	    for(File f : new File(filepath).listFiles()) { 
	        labels.add(f.getName());
	    }
	    
	    RecordReader recordReader = new ImageRecordReader(32, 32, 3, true, labels);
	    
	    recordReader.initialize(new FileSplit(new File(filepath)));
	    
	    DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 3072, 10);
    	
		List<DataSet> test = new ArrayList<>();
		
		while(iter.hasNext()){
			DataSet ds = iter.next();
			//System.out.println(ds);
			test.add(ds);
        }
		

        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson =
                MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("model/c_conf.json")));

        //Load parameters from disk:
        INDArray newParams;
        try (DataInputStream dis = new DataInputStream(new FileInputStream("model/c_coefficients.bin"))) {
            newParams = Nd4j.read(dis);
        }

        //Create a MultiLayerNetwork from the saved configuration and parameters
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParameters(newParams);

        //Evaluate (locally)
        Evaluation eval = new Evaluation();
        int maxid = -1;
        float max = -1;
        for (DataSet ds : test) {
        
            INDArray output = savedNetwork.output(ds.getFeatureMatrix());
            for(int i=0; i<output.columns(); i++){
            	if(max < output.getFloat(i)){
            		max = output.getFloat(i);
            		maxid = i;
            	}
            }
            //System.out.println(output);
        }
    	return maxid;
    }
}
