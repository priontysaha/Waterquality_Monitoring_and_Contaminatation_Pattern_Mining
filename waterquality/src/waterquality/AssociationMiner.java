package waterquality;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class AssociationMiner {

    public static void runApriori(Instances data) throws Exception {
        System.out.println("\n=== ASSOCIATION RULE MINING ===");

        Discretize discretize = new Discretize();
        discretize.setBins(3);
        discretize.setInputFormat(data);
        Instances discretizedData = Filter.useFilter(data, discretize);

        Apriori apriori = new Apriori();
        apriori.setNumRules(10);
        apriori.setMinMetric(0.70);
        apriori.setCar(true); // Class Association Rules
        apriori.buildAssociations(discretizedData);

        System.out.println(apriori);
    }
}