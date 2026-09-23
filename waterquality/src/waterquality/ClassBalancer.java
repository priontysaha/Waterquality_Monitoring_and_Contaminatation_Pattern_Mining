package waterquality;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.supervised.instance.SMOTE;

public class ClassBalancer {

    public static FilteredClassifier wrapWithSmote(Classifier baseClassifier, double percentage) throws Exception {
        SMOTE smote = new SMOTE();
        smote.setPercentage(percentage);
        smote.setNearestNeighbors(5);
        smote.setRandomSeed(1);

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(smote);
        fc.setClassifier(baseClassifier);
        return fc;
    }
}