package org.deeplearning4j.examples.cnn;

/**
 * Created by warn on 24/4/2016.
 */

import com.esotericsoftware.kryo.Kryo;
import org.apache.spark.serializer.KryoRegistrator;
import org.deeplearning4j.nn.updater.AdaGradUpdater;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;
import java.lang.reflect.Array;

/**
 * com.lordjoe.distributed.hydra.HydraKryoSerializer
 * User: Steve
 * Date: 10/28/2014
 */
public class HydraKryoSerializer implements KryoRegistrator, Serializable {

    /**
     * register a class
     *
     * @param kryo
     * @param pC   name of a class - might not exist
     */
    public void doRegistration(final Kryo kryo, final Class pC) {
        if (kryo != null) {
            kryo.register(pC);
            // also register arrays of that class
            Class arrayType = Array.newInstance(pC, 0).getClass();
            kryo.register(arrayType);
        }
    }

    /**
     * do the real work of registering all classes
     *
     * @param kryo
     */
    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(Object[].class);
        kryo.register(scala.Tuple2[].class);

        doRegistration(kryo, FloatBuffer.class);
        doRegistration(kryo, NDArray.class);
        doRegistration(kryo, AdaGrad.class);
        doRegistration(kryo, DataSet.class);
        doRegistration(kryo, AdaGradUpdater.class);
        doRegistration(kryo, MultiLayerUpdater.class);
    }
}

