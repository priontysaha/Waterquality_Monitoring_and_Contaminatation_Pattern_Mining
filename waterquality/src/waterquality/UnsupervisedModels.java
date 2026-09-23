package waterquality;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class UnsupervisedModels {

    public static void runClustering(Instances data, int k) throws Exception {
        System.out.println("\n=== UNSUPERVISED K-MEANS CLUSTERING ===");

        Remove removeClass = new Remove();
        removeClass.setAttributeIndices(String.valueOf(data.classIndex() + 1));
        removeClass.setInputFormat(data);
        Instances clusterData = Filter.useFilter(data, removeClass);

        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(k);
        kmeans.setSeed(1);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.buildClusterer(clusterData);

        System.out.println("K-Means SSE: " + kmeans.getSquaredError());

        int[] assignments = kmeans.getAssignments();
        int[][] crossTab = new int[k][2];

        for (int i = 0; i < data.numInstances(); i++) {
            int c = assignments[i];
            int cls = (int) data.instance(i).classValue();
            crossTab[c][cls]++;
        }

        System.out.println("\nCluster vs. Potability Breakdown:");
        System.out.printf("%-10s %-12s %-10s %-12s\n", "Cluster", "Unsafe (0)", "Safe (1)", "Unsafe Rate");
        for (int c = 0; c < k; c++) {
            int total = crossTab[c][0] + crossTab[c][1];
            double unsafeRate = (crossTab[c][0] * 100.0) / total;
            System.out.printf("Cluster %-2d %-12d %-10d %-10.2f%%\n", c, crossTab[c][0], crossTab[c][1], unsafeRate);
        }
    }
}