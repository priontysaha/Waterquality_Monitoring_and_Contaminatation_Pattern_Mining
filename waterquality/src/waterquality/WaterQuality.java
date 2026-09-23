package waterquality;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class WaterQuality {

    public static void main(String[] args) {
        try {
            String csvPath = (args.length > 0) ? args[0] : "NationalSurveyData.csv";
            String modelPath = "output/best_water_model.model";

            System.out.println("Starting Water Quality Analytics Pipeline...");

            // 1. Data Preparation
            Instances rawData = DataPreprocessor.loadAndParseCsv(csvPath, 50.0);
            Instances cleanData = DataPreprocessor.cleanAndFilter(rawData);

            // 2. Exploratory Data Analysis
            ExploratoryAnalyzer.printSummary(cleanData);

            // 3. Supervised Classification & SMOTE
            SupervisedModels.evaluateAll(cleanData);

            // 4. K-Means Clustering
            UnsupervisedModels.runClustering(cleanData, 4);

            // 5. Association Rule Mining
            AssociationMiner.runApriori(cleanData);

            // 6. Model Persistence & Deployment Verification
            var finalModel = ClassBalancer.wrapWithSmote(new RandomForest(), 200.0);
            finalModel.buildClassifier(cleanData);
            ModelPersistence.saveModel(modelPath, finalModel, new Instances(cleanData, 0));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}