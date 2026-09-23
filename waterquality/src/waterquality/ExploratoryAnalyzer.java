package waterquality;

import weka.core.Attribute;
import weka.core.Instances;

public class ExploratoryAnalyzer {

    public static void printSummary(Instances data) {
        System.out.println("\n=== EXPLORATORY DATA ANALYSIS ===");
        System.out.println("Total Instances: " + data.numInstances());
        System.out.println("Total Attributes: " + data.numAttributes());

        Attribute target = data.classAttribute();
        if (target != null && target.isNominal()) {
            System.out.println("\n--- Target Class Distribution (" + target.name() + ") ---");
            int[] counts = data.attributeStats(target.index()).nominalCounts;
            for (int i = 0; i < target.numValues(); i++) {
                double pct = (counts[i] * 100.0) / data.numInstances();
                System.out.printf("  Class %s (%s): %d (%.2f%%)\n", 
                        target.value(i), (target.value(i).equals("0") ? "Unsafe" : "Safe"), counts[i], pct);
            }
        }

        System.out.println("\n--- Chemical Parameter Summary Statistics ---");
        System.out.printf("%-15s %-10s %-10s %-10s %-10s\n", "Attribute", "Mean", "StdDev", "Min", "Max");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            if (att.isNumeric()) {
                double mean = data.meanOrMode(i);
                double std = Math.sqrt(data.variance(i));
                double min = data.attributeStats(i).numericStats.min;
                double max = data.attributeStats(i).numericStats.max;
                System.out.printf("%-15s %-10.2f %-10.2f %-10.2f %-10.2f\n", att.name(), mean, std, min, max);
            }
        }
    }
}