package waterquality;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class SupervisedModels {

    public static void evaluateAll(Instances data) throws Exception {
        System.out.println("\n=== SUPERVISED CLASSIFICATION EVALUATION ===");

        J48 j48 = new J48();
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);

        SMO smo = new SMO();
        smo.setKernel(new PolyKernel());
        smo.setBuildCalibrationModels(true);

        Classifier[] models = {j48, rf, smo};
        String[] modelNames = {"J48 Decision Tree", "Random Forest", "SMO (SVM)"};

        for (int i = 0; i < models.length; i++) {
            System.out.println("\n--- " + modelNames[i] + " (Standard) ---");
            runCV(models[i], data);

            System.out.println("\n--- " + modelNames[i] + " (With SMOTE) ---");
            Classifier smoteModel = ClassBalancer.wrapWithSmote(models[i], 200.0);
            runCV(smoteModel, data);
        }
    }

    private static void runCV(Classifier classifier, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));

        System.out.printf("Accuracy:           %.2f%%\n", eval.pctCorrect());
        System.out.printf("Weighted Precision: %.3f\n", eval.weightedPrecision());
        System.out.printf("Weighted Recall:    %.3f\n", eval.weightedRecall());
        System.out.printf("Weighted F1-Score:  %.3f\n", eval.weightedFMeasure());
        System.out.printf("ROC AUC:            %.3f\n", eval.weightedAreaUnderROC());
    }
}