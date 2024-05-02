package helper;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.security.Key;
import java.util.Map;
import java.util.Set;
import java.util.List;
import java.util.Random;
import java.util.HashSet;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class KMeansHelper {


    public static Map<Integer, Double> parse_term_tfidf(String line){

        Map<Integer, Double> term_tfidf = new HashMap<>();
        String[] term_tfidf_pairs = line.split(",");

        for (String pair : term_tfidf_pairs) {
            
            String[] parts = pair.split(":");
            int termId = Integer.parseInt(parts[0]);
            double tfidf = Double.parseDouble(parts[1]);
            // Add the term id and tfidf value to the map
            term_tfidf.put(termId, tfidf);
        }

        return term_tfidf;
    }

    
    public static double cosine_similarity (Map<Integer, Double> doc1, Map<Integer, Double> doc2){
        double dot_product = 0.0;

        for (Map.Entry<Integer, Double> entry : doc1.entrySet()) 
        {
            int term_id = entry.getKey();

            if (doc2.containsKey(term_id)){
                double tfidf1 = entry.getValue();
                double tfidf2 = doc2.get(term_id); 
                dot_product += tfidf1 * tfidf2;
            }   
        }

        double doc1_l2_length = 0.0;
        double doc2_l2_length = 0.0;

        for (double tfidf : doc1.values()) {
            doc1_l2_length += tfidf * tfidf;
        }
        for (double tfidf : doc2.values()) {
            doc2_l2_length += tfidf * tfidf;
        }
        doc1_l2_length = Math.sqrt(doc1_l2_length);
        doc2_l2_length = Math.sqrt(doc2_l2_length);

        if (doc1_l2_length == 0.0 || doc2_l2_length == 0.0) {
            return 0.0; 
        }
        return Math.round(dot_product * 1000.0 / (doc1_l2_length * doc2_l2_length))/1000.0;
    }

    // Find sum squares for a cluster.
    public static double sum_squares(Map<Integer, Double> term_tfidf, Map<Integer, Double> centroid) 
    {
        double sum_of_squares = 0.0;

        Set<Integer> keys_union = new HashSet<>();
        keys_union.addAll(term_tfidf.keySet());
        keys_union.addAll(centroid.keySet());
      
        for (int term_id : keys_union) {

            double tfidf_1 = term_tfidf.getOrDefault(term_id, 0.0);
            double tfidf_2 = centroid.getOrDefault(term_id, 0.0);
            double diff = tfidf_1 - tfidf_2;
            sum_of_squares += diff * diff;
        }

        return sum_of_squares;
    }
    
}
