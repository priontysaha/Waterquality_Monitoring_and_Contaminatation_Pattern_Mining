package waterquality;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PotabilityPredictor {

    public static void predict(String modelPath, double[] sampleFeatures) throws Exception {
        System.out.println("\n=== REAL-TIME POTABILITY PREDICTION ===");
        Object[] loaded = ModelPersistence.loadModel(modelPath);
        Classifier model = (Classifier) loaded[0];
        Instances header = (Instances) loaded[1];

        Instance inst = new DenseInstance(1.0, sampleFeatures);
        inst.setDataset(header);

        double classVal = model.classifyInstance(inst);
        double[] probs = model.distributionForInstance(inst);

        String result = header.classAttribute().value((int) classVal);
        System.out.printf("Prediction: %s (%s)\n", result, result.equals("0") ? "UNSAFE" : "SAFE");
        System.out.printf("Probabilities -> Unsafe: %.2f%% | Safe: %.2f%%\n", probs[0] * 100, probs[1] * 100);
    }
}