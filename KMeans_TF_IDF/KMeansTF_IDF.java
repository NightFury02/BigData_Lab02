import java.io.IOException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;

import java.util.Map;
import java.util.Set;
import java.util.List;
import java.util.Random;
import java.util.HashSet;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;


public class KMeansTF_IDF {

    public static ArrayList<String> centroids = new ArrayList<>();
    public static ArrayList<Double> iter_loss = new ArrayList<>();
    public static ArrayList<String> top_terms = new ArrayList<>();

    // Function to get k random centroids
    public static void init_random_centroids(int k, String input_path) throws IOException
    {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path input_file_path = new Path("hdfs://localhost:9000" + input_path);

        if (fs.exists(input_file_path)) {
            ArrayList<String> lines = new ArrayList<>();

            try (FSDataInputStream input_stream = fs.open(input_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
                String line;

                while ((line = reader.readLine()) != null) {
                    lines.add(line);
                }

                Collections.shuffle(lines);

                for (int i = 0; i < k; i++)
                {
                    line = lines.get(i);
                    String[] parts = line.split("\\|");
                    centroids.add(parts[1].trim());   
                }
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }

            for (int i = 0; i < centroids.size(); i++){
                System.out.println(centroids.get(i));
            }
        }
    }

    public static String centroids_to_string()
    {
        StringBuilder centroids_string = new StringBuilder();

        for (String centroid : centroids) {
            centroids_string.append(centroid).append(";");
        }
        
        return centroids_string.toString();
    }

    private static void read_centroids(String centroid_path, int k) throws IOException {

        System.out.println("*******READ CENTROID FUNCTION**********");
        
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // Path to centroids file in HDFS
        Path centroid_file_path = new Path("hdfs://localhost:9000"+ centroid_path);

        if (fs.exists(centroid_file_path)) {
            // Clear all current centroids
            centroids.clear();

            try (FSDataInputStream input_stream = fs.open(centroid_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    centroids.add(parts[0]);
                }

                System.out.println("CENTROID SIZE: "+ centroids.size());
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Centroid file doesn't exist in HDFS.");
        }
        System.out.println("********FINISH READING CENTROIDS**********");
    }

    public static void convert_file(String input_path, String output_path) throws IOException{
        System.out.println("**********CONVERT INPUT FILE FUNCTION************");

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path input_file_path = new Path("hdfs://localhost:9000" + input_path + "/input.mtx");
        Path output_file_path = new Path("hdfs://localhost:9000" + output_path);

        if (fs.exists(input_file_path)) {
            FSDataOutputStream output_stream = fs.create(output_file_path);
            StringBuilder output_builder = new StringBuilder();

            try (FSDataInputStream input_stream = fs.open(input_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;
                // Skip the descriptive line and matrix-size line.
                line = reader.readLine();

                while (line != null && line.startsWith("%")) 
                {
                    line = reader.readLine();
                }

                // A map to store doc_id, (term_id, tf_idf)
                Map<Integer, Map<Integer, Double>> doc_term_map = new HashMap<>();

                while ((line = reader.readLine()) != null ) {

                    String[] parts = line.split("\\s+");
                    int doc_id = Integer.parseInt(parts[1]);
                    int term_id = Integer.parseInt(parts[0]);
                    double tf_idf = Double.parseDouble(parts[2]);

                    if (!doc_term_map.containsKey(doc_id)) {
                        doc_term_map.put(doc_id, new HashMap<>());
                    }

                    doc_term_map.get(doc_id).put(term_id, tf_idf);
                }

                // Build new input file
                for (Map.Entry<Integer, Map<Integer, Double>> entry : doc_term_map.entrySet()) 
                {
                    int doc_id = entry.getKey();
                    Map<Integer, Double> term_tfidf_map = entry.getValue();
                    
                    output_builder.append(doc_id).append("|");

                    for (Map.Entry<Integer, Double> term_tfidf : term_tfidf_map.entrySet())
                    {
                        int term_id = term_tfidf.getKey();
                        double tf_idf = term_tfidf.getValue();
                        output_builder.append(term_id).append(":").append(tf_idf).append(",");
                    }
                    
                    // Remove the last comma and add a newline character
                    output_builder.deleteCharAt(output_builder.length() - 1);
                    output_builder.append("\n");
                }
                output_stream.write(output_builder.toString().getBytes());
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }finally {
                try {
                    if (output_stream != null) {
                        output_stream.close();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        System.out.println("*************FINISH CONVERTING FILE**********");
    }

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

    public static boolean has_converged(ArrayList<String> old_centroids, ArrayList<String> new_centroids, double threshold) {
        System.out.println("********CHECKING CONVERGENGE FUNCTION*********");

        // Check if centroids have converged based on a threshold
        boolean result = true;

        for (int i = 0; i < old_centroids.size(); i++) {

            Map<Integer, Double> old_centroid = parse_term_tfidf(old_centroids.get(i));
            Map<Integer, Double> new_centroid = parse_term_tfidf(new_centroids.get(i));
           

            double similarity = cosine_similarity(old_centroid, new_centroid);

            System.out.println("Similarity " + i + ": " + similarity);

            if (similarity < threshold) {
                result = false;
            }
        }

        System.out.println("********CHECKED CONVERGENGE*********");
        if (result == true){
            System.out.println("Centroids have converged.");
        }else{
            System.out.println("Centroids are still changing.");
        }
        return result;
    }

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text>{

        public static ArrayList<Map<Integer, Double>> mapper_centroids = new  ArrayList<>();
 
        public void setup(Context context) throws IOException, InterruptedException{

            // Get centroids string from Kmeans class to Mapper class
            Configuration conf = context.getConfiguration();
            String centroids_string = conf.get("centroids");
            String[] centroids_list = centroids_string.split(";");

            for (String centroid : centroids_list){

                Map<Integer, Double> term_tfidf = new HashMap<>();
                String[] term_tfidf_pairs = centroid.split(",");
        
                for (String pair : term_tfidf_pairs) {
                    
                    String[] parts = pair.split(":");
                    int term_id = Integer.parseInt(parts[0]);
                    double tfidf = Double.parseDouble(parts[1]);
                    // Add the term id and tfidf value to the map
                    term_tfidf.put(term_id, tfidf);
                }
                mapper_centroids.add(term_tfidf);
            }
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] doc_term_tfidf = value.toString().split("\\|");

            int doc_id = Integer.parseInt(doc_term_tfidf[0]);
            Map<Integer, Double> term_tfidf = KMeansTF_IDF.parse_term_tfidf(doc_term_tfidf[1]);
                
            double max_similarity = -1;
            int cluster = -1;
           
           for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double similarity =   KMeansTF_IDF.cosine_similarity(term_tfidf, centroid);

                if (similarity > max_similarity) {
                    cluster = cluster_id;
                    max_similarity = similarity;
                }
            }
            context.write(new IntWritable(cluster), value);
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Text, Text, IntWritable> {
    
        private MultipleOutputs<Text, IntWritable > mos;
        private ArrayList<Map<Integer, Double>>  line_values = new ArrayList<>();
        
        public void setup(Context context) throws IOException, InterruptedException {
            mos = new MultipleOutputs<>(context);
        }

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Map<Integer, Double> new_centroid = new HashMap<>();
            int num_values = 0;
            
            for (Text value: values){
                String[] doc_term_tfidf = value.toString().split("\\|");

                int doc_id = Integer.parseInt(doc_term_tfidf[0]);
                Map<Integer, Double> term_tfidf = KMeansTF_IDF.parse_term_tfidf(doc_term_tfidf[1]);
                line_values.add(term_tfidf);
                
                // Find the sum of tfidf in cluster
                for (Map.Entry<Integer, Double> entry : term_tfidf.entrySet()) {
                    int term_id = entry.getKey();
                    double tfidf = entry.getValue();
    
                    new_centroid.put(term_id, new_centroid.getOrDefault(term_id, 0.0) + tfidf);
                }
                num_values++;
                mos.write("cluster", new Text(Integer.toString(doc_id)), key, "task_2_2.classes");
            }

            List<Map.Entry<Integer, Double>> sorted_entries = new ArrayList<Map.Entry<Integer, Double>>(new_centroid.entrySet());
            sorted_entries.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
            List<Map.Entry<Integer, Double>> top_10_entries = sorted_entries.subList(0, Math.min(10, sorted_entries.size()));

            StringBuilder top_10_terms = new StringBuilder();

            for (Map.Entry<Integer, Double> entry : top_10_entries) {
                int term_id = entry.getKey();
                double tfidf = entry.getValue();
                top_10_terms.append(term_id).append(":").append(tfidf).append(",");
            }
            top_10_terms.deleteCharAt(top_10_terms.length() - 1);
            mos.write("topterms", new Text(top_10_terms.toString()), key, "task_2_2.topterms");

            StringBuilder new_centroid_builder = new StringBuilder();

            for (Map.Entry<Integer, Double> entry : new_centroid.entrySet()) {
                int term_id = entry.getKey();
                double sum_tfidf = entry.getValue();
    
                double avg_tfidf = sum_tfidf / num_values;

                new_centroid.put(term_id, avg_tfidf);

                new_centroid_builder.append(term_id).append(":").append(avg_tfidf).append(",");
            }
            new_centroid_builder.deleteCharAt(new_centroid_builder.length() - 1);
            mos.write("centroid", new Text(new_centroid_builder.toString()), key, "task_2_2.clusters");
            
            double sum_squared_distances = 0.0;
        
            for (int i = 0; i < line_values.size(); i++) {
                Map<Integer, Double> term_tfidf = line_values.get(i);
                sum_squared_distances += KMeansTF_IDF.sum_squares(term_tfidf, new_centroid);
            }
     
            mos.write("loss",new Text(Double.toString(sum_squared_distances)), key, "cluster.loss");
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static void rename_output_files(String output_dir) throws IOException
    {

        System.out.println("********RENAME OUTPUT FILE FUNCTION*********");

        FileSystem hdfs = FileSystem.get(new Configuration());
        Path output_path = new Path(output_dir);
        FileStatus fs[] = hdfs.listStatus(output_path);

        if (fs != null) {
            for (FileStatus file : fs) 
            {
                // If file is not a directory, remove the suffix
                if (!file.isDir()) {
                    String file_name = file.getPath().getName();
                    int index = file_name.indexOf("-r-00000");

                    if (index != -1) {
                        String new_file_name = file_name.substring(0, index);
                        Path new_path = new Path("hdfs://localhost:9000"+ output_dir + "/" + new_file_name);
                        
                        hdfs.rename(file.getPath(), new_path);
                    }
                }
            }
        }
    }

    public static void get_top_terms (int iteration, String file_path) throws IOException{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path("hdfs://localhost:9000"+ file_path + "/task_2_2.topterms");
    
        if (fs.exists(loss_file_path)) {

            try (FSDataInputStream input_stream = fs.open(loss_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    top_terms.add(line);
                }
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Top term file doesn't exist in HDFS.");
        }
    }

    public static void write_top_terms (int k, String file_path) throws IOException{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path output_file_path = new Path("hdfs://localhost:9000" + file_path + "/task_2_2.txt");

        if (!fs.exists(output_file_path)) {
            fs.createNewFile(output_file_path);
        }
    
        int count  = 0;
        int iter = 0;
        try (FSDataOutputStream output_stream = fs.append(output_file_path)) {
            for (int i = 0; i < top_terms.size(); i++)
            {
                if(count == k){
                    count = 0;
                }
                if (count == 0){
                    String iteration = "Iteration: " + iter + "\n";
                    output_stream.write(iteration.getBytes());
                    iter++;
                }
               
                String terms = top_terms.get(i) + "\n";
                output_stream.write(terms.getBytes());
                count++;   
            }
        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
        }
    }

    public static void calculate_loss (int iteration, String file_path) throws IOException
    {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path("hdfs://localhost:9000"+ file_path + "/cluster.loss");
        
        Double loss = 0.0;

        if (fs.exists(loss_file_path)) {

            try (FSDataInputStream input_stream = fs.open(loss_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                 
                    String[] parts = line.trim().split("\\s+");
                    loss += Double.parseDouble(parts[0]);
                }

                iter_loss.add(loss);
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Loss file doesn't exist in HDFS.");
        }
    }

    public static void write_loss(String file_path) throws IOException{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path output_file_path = new Path("hdfs://localhost:9000" + file_path + "/task_2_2.loss");

        if (!fs.exists(output_file_path)) {
            fs.createNewFile(output_file_path);
        }
    
        try (FSDataOutputStream output_stream = fs.append(output_file_path)) {
            for (int i = 0; i < iter_loss.size(); i++){
                String output_line = "Iteration: " + i + ", Loss = " + iter_loss.get(i) + "\n";
                output_stream.write(output_line.getBytes());
            }
           
        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
        }
    }

    public static void run_map_reduce_job(int iteration, int k, String input_file, String output_file) throws Exception{

        Configuration conf = new Configuration();

        // Convert the centroid to string and set a new congiuration parameter.
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "KMeans Clustering");

        job.setJarByClass(KMeansTF_IDF.class);
        
        // Config Map phase
        job.setMapperClass(KMeansMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);	

        // Config Reduce phase.
        job.setReducerClass(KMeansReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        // If output path has already existed, delete it before a job.
        Path output_path = new Path(output_file);
        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }
	
        FileInputFormat.setInputPaths(job, new Path(input_file));
        FileOutputFormat.setOutputPath(job, output_path);
        MultipleOutputs.addNamedOutput(job, "centroid", TextOutputFormat.class,  Text.class, IntWritable.class);
        MultipleOutputs.addNamedOutput(job, "cluster", TextOutputFormat.class, Text.class, IntWritable.class);
        MultipleOutputs.addNamedOutput(job, "topterms", TextOutputFormat.class, Text.class, IntWritable.class);
        MultipleOutputs.addNamedOutput(job, "loss", TextOutputFormat.class, Text.class, IntWritable.class);

        if (job.waitForCompletion(true)) {
            // If job completed, rename the output file. 
            rename_output_files(output_file);

        } else {
            System.out.println("MAP REDUCE JOB FAIL");
        }
    }

    // Run an interation
    public static void run(int iteration, int k, String input_file, String output_file) throws Exception {

        if (iteration == 0) {
            // Generate random centroids and write them to file
            init_random_centroids(k, input_file);
        }

        try{
            run_map_reduce_job( iteration, k, input_file, output_file);
            read_centroids(output_file + "/task_2_2.clusters", k);
            calculate_loss(iteration, output_file);
            get_top_terms(iteration, output_file);
            
        } catch (Exception e){
            System.out.println("Exception: " + e.getMessage());
            System.out.println("ERROR");
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println("Invalid parameters. Use: <input_path> <output_path> <k> <max_iterations>");
            System.exit(1);
        }

        String input_file = args[0];
        String output_file = args[1];
        int k = Integer.parseInt(args[2]);
        int max_iterations = Integer.parseInt(args[3]);

        ArrayList<String> old_centroids = new  ArrayList<>();
        ArrayList<String> new_centroids = new  ArrayList<>();

        // Convert the input file to a desired format.
        String converted_input_file = input_file + "/tfidf.txt";
        convert_file(input_file, converted_input_file );

        int iteration = 0;

        while (iteration < max_iterations)
        {
            System.out.println("+++++++++++++++++");
            System.out.println("Iteration: " + iteration);

            if (iteration > 0){
                old_centroids = new  ArrayList<>(centroids);
            }

            run(iteration, k, converted_input_file, output_file);

            if (iteration > 0){
                new_centroids = new  ArrayList<>(centroids);

                if (has_converged(old_centroids, new_centroids,0.95) == true) {
                    break;
                }
            }
            iteration ++;
        }

        write_top_terms(k, output_file);
        write_loss(output_file);

    }
}
