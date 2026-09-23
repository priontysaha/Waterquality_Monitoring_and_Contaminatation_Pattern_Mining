package waterquality;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;

public class DataPreprocessor {

    public static Instances loadAndParseCsv(String csvPath, double asThreshold) throws IOException {
        File file = new File(csvPath);
        if (!file.exists()) throw new IOException("CSV file not found: " + csvPath);

        List<String[]> rows = new ArrayList<>();
        String[] headers = null;

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            int lineNum = 0;
            while ((line = br.readLine()) != null) {
                lineNum++;
                if (lineNum <= 4) continue; // Skip title metadata

                String[] tokens = parseCsvLine(line);
                if (lineNum == 5) { headers = tokens; continue; }
                if (lineNum == 6) continue; // Skip units row

                if (tokens.length > 0 && !tokens[0].trim().isEmpty()) {
                    rows.add(tokens);
                }
            }
        }
        return createInstances(headers, rows, asThreshold);
    }

    private static Instances createInstances(String[] headers, List<String[]> rows, double threshold) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        int asIdx = -1;

        for (int i = 0; i < headers.length; i++) {
            String h = headers[i].trim();
            if (h.equalsIgnoreCase("As")) asIdx = i;

            if (h.equalsIgnoreCase("WELL_TYPE") || h.equalsIgnoreCase("DIVISION") || 
                h.equalsIgnoreCase("DISTRICT") || h.equalsIgnoreCase("THANA") || 
                h.equalsIgnoreCase("UNION") || h.equalsIgnoreCase("MOUZA") || 
                h.equalsIgnoreCase("SAMPLE_ID") || h.equalsIgnoreCase("SAMPLE_FIELD_ID") || 
                h.equalsIgnoreCase("SAMPLE_DATE")) {
                attributes.add(new Attribute(h, (List<String>) null));
            } else {
                attributes.add(new Attribute(h));
            }
        }

        Attribute potabilityAttr = new Attribute("Potability", Arrays.asList("0", "1"));
        attributes.add(potabilityAttr);

        Instances dataset = new Instances("WaterQuality", attributes, rows.size());
        dataset.setClassIndex(dataset.numAttributes() - 1);

        for (String[] row : rows) {
            Instance inst = new DenseInstance(attributes.size());
            inst.setDataset(dataset);
            double asValue = Double.NaN;

            for (int i = 0; i < headers.length && i < row.length; i++) {
                String val = row[i] != null ? row[i].trim() : "";
                Attribute att = attributes.get(i);

                if (val.isEmpty() || val.equalsIgnoreCase("NA")) {
                    inst.setMissing(att);
                    continue;
                }

                if (att.isString()) {
                    inst.setValue(att, val);
                } else if (att.isNumeric()) {
                    double parsed = parseCensored(val);
                    if (Double.isNaN(parsed)) {
                        inst.setMissing(att);
                    } else {
                        inst.setValue(att, parsed);
                        if (i == asIdx) asValue = parsed;
                    }
                }
            }

            if (Double.isNaN(asValue)) {
                inst.setMissing(potabilityAttr);
            } else {
                inst.setValue(potabilityAttr, (asValue > threshold) ? "0" : "1");
            }
            dataset.add(inst);
        }
        return dataset;
    }

    private static double parseCensored(String val) {
        if (val.startsWith("<")) {
            try { return Double.parseDouble(val.substring(1).trim()) / 2.0; } catch (Exception e) { return Double.NaN; }
        } else if (val.startsWith(">")) {
            try { return Double.parseDouble(val.substring(1).trim()); } catch (Exception e) { return Double.NaN; }
        } else {
            try { return Double.parseDouble(val); } catch (Exception e) { return Double.NaN; }
        }
    }

    private static String[] parseCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        boolean inQuotes = false;
        StringBuilder sb = new StringBuilder();
        for (char c : line.toCharArray()) {
            if (c == '"') inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes) {
                tokens.add(sb.toString().trim());
                sb.setLength(0);
            } else sb.append(c);
        }
        tokens.add(sb.toString().trim());
        return tokens.toArray(new String[0]);
    }

    public static Instances cleanAndFilter(Instances dataset) throws Exception {
        StringToNominal stn = new StringToNominal();
        stn.setAttributeRange("first-last");
        stn.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, stn);

        List<Integer> indicesToRemove = new ArrayList<>();
        String[] leakageNames = {"SAMPLE_ID", "SAMPLE_FIELD_ID", "SAMPLE_DATE", "LAT_DEG", "LONG_DEG", 
                                 "DIVISION", "DISTRICT", "THANA", "UNION", "MOUZA", "GEOCODE", "As"};
        for (String name : leakageNames) {
            Attribute att = dataset.attribute(name);
            if (att != null) indicesToRemove.add(att.index());
        }

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove.stream().mapToInt(i -> i).toArray());
        remove.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, remove);

        ReplaceMissingValues impute = new ReplaceMissingValues();
        impute.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, impute);

        Normalize norm = new Normalize();
        norm.setInputFormat(dataset);
        return Filter.useFilter(dataset, norm);
    }
}
