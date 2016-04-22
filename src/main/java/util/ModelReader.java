package org.deeplearning4j.examples.mlp;

import org.apache.commons.io.FileUtils;
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
public class ModelReader {

    private static final Logger log = LoggerFactory.getLogger(ModelReader.class);

    public static void main(String[] args) throws Exception {
    	
    	
    	int label = modelReader("input.txt");
    	System.out.println("Label is " + label);
    	
    }
    
    static int modelReader(String file)throws Exception{
    	
		float[] input = new float[784];
		float[] out = new float[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		BufferedReader bf = new BufferedReader(new FileReader( new File(file) ));
		String[] s = bf.readLine().split(",");
		for(int i=0; i<s.length; i++) input[i] = Float.valueOf(s[i]);
		
		INDArray array1 = Nd4j.create(input);
		INDArray array2 = Nd4j.create(out);
		DataSet inds = new DataSet(array1, array2);
		List<DataSet> test = new ArrayList<>(1);
		test.add(inds);

        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson =
                MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));

        //Load parameters from disk:
        INDArray newParams;
        try (DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"))) {
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
        }
    	return maxid;
    }
}
