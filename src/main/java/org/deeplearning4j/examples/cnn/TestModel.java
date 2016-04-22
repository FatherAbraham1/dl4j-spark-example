package org.deeplearning4j.examples.cnn;

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
public class TestModel {

    private static final Logger log = LoggerFactory.getLogger(TestModel.class);

    public static void main(String[] args) throws Exception {

        //------Simple Dataset--------
        int numSamples = 100;

        log.info("Load data....");
        DataSetIterator mnistIter = new MnistDataSetIterator(1, numSamples, true);
        List<DataSet> allData = new ArrayList<>(numSamples);
        while (mnistIter.hasNext()) {
            allData.add(mnistIter.next());
        }
        Collections.shuffle(allData, new Random(12345));

        Iterator<DataSet> iter = allData.iterator();
        List<DataSet> test = new ArrayList<>(numSamples);

        while (iter.hasNext()) {
            test.add(iter.next());
        }

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
        log.info(eval.stats());
    }
}
