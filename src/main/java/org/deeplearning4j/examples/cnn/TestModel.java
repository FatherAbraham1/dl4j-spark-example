package org.deeplearning4j.examples.cnn;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class TestModel {
	
    public static void main(String[] args) throws Exception {
	
	//------Simple Dataset--------
        List<DataSet> test = new ArrayList<>(1);
        BufferedReader bf = new BufferedReader(new FileReader(new File("input.txt")));
        float[] input = new float[784];
        String[] s = bf.readLine().split(",");
        for (int i = 0; i < s.length; i++) input[i] = Float.valueOf(s[i]);

        INDArray array1 = Nd4j.create(input);
        INDArray array2 = Nd4j.create(new int[]{1});
        DataSet inds = new DataSet(array1, array2);
        test.add(inds);

        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson =
                MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("model/conf.json")));
        
        //Load parameters from disk:
        INDArray newParams;
        try (DataInputStream dis = new DataInputStream(new FileInputStream("model/coefficients.bin"))) {
            newParams = Nd4j.read(dis);
        }

        //Create a MultiLayerNetwork from the saved configuration and parameters
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParameters(newParams);

        //Evaluate (locally)
        Evaluation eval = new Evaluation();
        for (DataSet ds : test) {
            INDArray output = savedNetwork.output(ds.getFeatureMatrix());
            System.out.println("**" + output);
            eval.eval(ds.getLabels(), output);
        }
    }
}
