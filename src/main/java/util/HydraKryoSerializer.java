package util;

/**
 * Created by warn on 24/4/2016.
 */

import com.esotericsoftware.kryo.Kryo;
import org.apache.spark.serializer.KryoRegistrator;

import javax.annotation.Nonnull;
import java.io.Serializable;
import java.lang.reflect.Array;

/**
 * com.lordjoe.distributed.hydra.HydraKryoSerializer
 * User: Steve
 * Date: 10/28/2014
 */
public class HydraKryoSerializer implements KryoRegistrator, Serializable {
    public HydraKryoSerializer() {
    }

    /**
     * register a class indicated by name
     *
     * @param kryo
     * @param s       name of a class - might not exist
     */
    protected void doRegistration(@Nonnull Kryo kryo, @Nonnull String s) {
        Class c;
        try {
            c = Class.forName(s);
            doRegistration(kryo, c);
        } catch (ClassNotFoundException e) {
            return;
        }
    }

    /**
     * register a class
     *
     * @param kryo
     * @param pC       name of a class - might not exist
     */
    protected void doRegistration(final Kryo kryo, final Class pC) {
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
    public void registerClasses(@Nonnull Kryo kryo) {
        kryo.register(Object[].class);
        kryo.register(scala.Tuple2[].class);

        doRegistration(kryo, "org.nd4j.linalg.api.buffer.FloatBuffer");
        doRegistration(kryo, "org.nd4j.linalg.cpu.NDArray");
        doRegistration(kryo, "org.nd4j.linalg.learning.AdaGrad");
        doRegistration(kryo, "org.deeplearning4j.nn.updater.AdaGradUpdate");
        doRegistration(kryo, "org.deeplearning4j.nn.updater.MultiLayerUpdater");
    }
}

