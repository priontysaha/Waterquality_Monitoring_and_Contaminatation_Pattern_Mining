package waterquality;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class ModelPersistence {

    public static void saveModel(String path, Classifier model, Instances header) throws Exception {
        SerializationHelper.writeAll(path, new Object[]{model, header});
        System.out.println("Model saved successfully to: " + path);
    }

    public static Object[] loadModel(String path) throws Exception {
        Object[] loaded = SerializationHelper.readAll(path);
        System.out.println("Model loaded successfully from: " + path);
        return loaded;
    }
}