package util;

import com.esotericsoftware.kryo.Kryo;
import de.javakaffee.kryoserializers.SynchronizedCollectionsSerializer;
import org.apache.spark.serializer.KryoRegistrator;
import org.apache.spark.util.StatCounter;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.AdaGradUpdater;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.NesterovsUpdater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.Nesterovs;
import scala.collection.mutable.WrappedArray;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * util.HydraKryoSerializer
 * User: Mark Wang
 * Date: 04/05/2016
 */
public class HydraKryoSerializer implements KryoRegistrator, Serializable {

    /**
     * register a class
     *
     * @param kryo kryo object
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
     * @param kryo kryo object
     */
    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(Object[].class);
        kryo.register(scala.Tuple2[].class);
        kryo.register(LinkedHashMap.class);
        kryo.register(NesterovsUpdater.class);
        kryo.register(Nesterovs.class);
        kryo.register(DataBuffer.class);
        kryo.register(DataBuffer.AllocationMode.class);
        kryo.register(AtomicBoolean.class);
        kryo.register(double[].class);
        kryo.register(float[].class);
        kryo.register(int[].class);
        kryo.register(boolean[].class);
        kryo.register(StatCounter.class);
        kryo.register(AdaGrad.AdaGradAggregator.class);
        try {
            kryo.register(Class.forName("org.deeplearning4j.nn.updater.MultiLayerUpdater$MultiLayerUpdaterAggregator"));
            kryo.register(Class.forName("org.deeplearning4j.nn.updater.NesterovsUpdater$NesterovsAggregator"));
            kryo.register(Class.forName("org.nd4j.linalg.learning.Nesterovs$NesterovsAggregator"));
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        SynchronizedCollectionsSerializer.registerSerializers(kryo);

        doRegistration(kryo, FloatBuffer.class);
        doRegistration(kryo, NDArray.class);
        doRegistration(kryo, AdaGrad.class);
        doRegistration(kryo, DataSet.class);
        doRegistration(kryo, AdaGradUpdater.class);
        doRegistration(kryo, MultiLayerUpdater.class);
        doRegistration(kryo, Updater.class);
        doRegistration(kryo, HashSet.class);
        doRegistration(kryo, WrappedArray.ofRef.class);
        doRegistration(kryo, ArrayList.class);
        doRegistration(kryo, UpdaterAggregator.class);
    }
}

