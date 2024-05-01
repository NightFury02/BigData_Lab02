import java.io.IOException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.security.Key;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.List;
import java.util.Random;
import java.util.HashSet;
import java.util.HashMap;
import java.util.AbstractMap;
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

public class KMeans_ll
{
    // public static ArrayList<String> centroids = new ArrayList<>();
    public static Set<String> centroids = new HashSet<>();

    public static ArrayList<String> init_random_centroid(int k, String input_path) throws IOException
    {
        ArrayList<String> centroids = new ArrayList<>();
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

                for (int i = 0; i < k; i++){
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

        return centroids;
    }

    public static String centroids_to_string()
    {
        StringBuilder centroids_string = new StringBuilder();

        for (String centroid : centroids) {
            centroids_string.append(centroid).append(";");
        }
        
        centroids_string.deleteCharAt(centroids_string.length() - 1);
        return centroids_string.toString();
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

    public static void rename_output_files(String output_dir) throws IOException
    {
        FileSystem hdfs = FileSystem.get(new Configuration());
        Path output_path = new Path(output_dir);
        FileStatus fs[] = hdfs.listStatus(output_path);

        if (fs != null) {
            for (FileStatus file : fs) 
            {
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

    public static double sum_squares(Map<Integer, Double> term_tfidf, Map<Integer, Double> centroid) 
    {
        double sum_of_squares = 0.0;

        for (int term_id : term_tfidf.keySet()) {
            double tfidf_1 = term_tfidf.get(term_id);
            double tfidf_2 = centroid.getOrDefault(term_id, 0.0);
            double diff = tfidf_1 - tfidf_2;
            sum_of_squares += diff * diff;
        }
    
        // Iterate through the keys of centroid that are not in term_tfidf
        for (int term_id : centroid.keySet()) {
            if (!term_tfidf.containsKey(term_id)) {
                double tfidf_1 = 0.0; // term_tfidf default value is 0.0
                double tfidf_2 = centroid.get(term_id);
                double diff = tfidf_1 - tfidf_2;
                sum_of_squares += diff * diff;
            }
        }
        return sum_of_squares;
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

    public static class CostMapper extends Mapper<Object, Text, IntWritable, DoubleWritable>{

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
            Map<Integer, Double> term_tfidf = parse_term_tfidf(doc_term_tfidf[1]);
            double min_distance = Double.MAX_VALUE;
            int cluster = -1;
           
           for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double distance =   KMeans_ll.sum_squares(term_tfidf, centroid);
                if (distance < min_distance){
                    min_distance = distance;
                    cluster = cluster_id;
                }

            }
            context.write(new IntWritable(cluster), new DoubleWritable( min_distance));
        }
    }  

    public static class CostReducer extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
        private MultipleOutputs<IntWritable, DoubleWritable > mos;

        public void setup(Context context) throws IOException, InterruptedException {
            mos = new MultipleOutputs<>(context);
        }

        public void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context) 
            throws IOException, InterruptedException {
            
            double sum = 0.0;
            for (DoubleWritable value : values) {
                sum += value.get();
            }
    
            mos.write("distance", key, new DoubleWritable(sum), "distances.txt");
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static class ProbabilityMapper extends Mapper<Object, Text, Text, DoubleWritable>{
        public static ArrayList<Map<Integer, Double>> mapper_centroids = new  ArrayList<>();
        private Double total_loss ;
        private Double over_sampling;

        protected void setup(Context context) throws IOException, InterruptedException {
            // Load centroids from context
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
                    term_tfidf.put(term_id, tfidf);
                }
                mapper_centroids.add(term_tfidf);
            }

            String loss_str = conf.get("loss");
            total_loss = Double.parseDouble(loss_str);

            String l = conf.get("l");
            over_sampling = Double.parseDouble(l);
            
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] doc_term_tfidf = value.toString().split("\\|");

            int doc_id = Integer.parseInt(doc_term_tfidf[0]);
            Map<Integer, Double> term_tfidf = parse_term_tfidf(doc_term_tfidf[1]);
            double min_distance = Double.MAX_VALUE;
            int cluster = -1;
           
           for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double distance =   KMeans_ll.sum_squares(term_tfidf, centroid);
                if (distance < min_distance){
                    min_distance = distance;
                }
            }
            Double probability = over_sampling * min_distance / total_loss;

            context.write(value, new DoubleWritable( probability));
        }
    }

    public static class ProbabilityReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        private Map<Text, Double> candidate_centroids = new HashMap<>();
        // private PriorityQueue<Map.Entry<Text, Double>> priorityQueue;
        private Double over_sampling;
        private MultipleOutputs<Text, DoubleWritable > mos;
      

        public void setup(Context context) throws IOException, InterruptedException {
         
            mos = new MultipleOutputs<>(context);
            Configuration conf = context.getConfiguration();
            String l = conf.get("l");
            over_sampling = Double.parseDouble(l);
            // priorityQueue = new PriorityQueue<>((a, b) -> Double.compare(b.getValue(), a.getValue()));
        }
        
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double max_probability = Double.MIN_VALUE;
            for (DoubleWritable value : values) {
                if (value.get() > max_probability){
                    max_probability = value.get();
                }
            }

            candidate_centroids.put(new Text(key), max_probability);
            // mos.write("probability", key, new DoubleWritable(max_probability), "probability.txt");

            // priorityQueue.offer(new AbstractMap.SimpleEntry<>(new Text(key), max_probability));
            // if (priorityQueue.size() > over_sampling) {
            //     priorityQueue.poll(); // Remove the entry with the lowest probability
            // }
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            List<Map.Entry<Text, Double>> sorted_entries = new ArrayList<>(candidate_centroids.entrySet());
            sorted_entries.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));

            int count = 1;
            for (Map.Entry<Text, Double> entry : sorted_entries) {
                if (count >= over_sampling) {
                    break;
                }
                mos.write("probability", entry.getKey(), new DoubleWritable(entry.getValue()), "probability.txt");
                count++;
            }

            // while (!priorityQueue.isEmpty()) {
            //     Map.Entry<Text, Double> entry = priorityQueue.poll();
            //     mos.write("probability", entry.getKey(), new DoubleWritable(entry.getValue()), "probability.txt");
            // }
            mos.close();
        }
    }

    public static Double calculate_loss (String file_path) throws IOException
    {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path("hdfs://localhost:9000"+ file_path + "/distances.txt");
        
        Double loss = 0.0;

        if (fs.exists(loss_file_path)) {

            try (FSDataInputStream input_stream = fs.open(loss_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                 
                    String[] parts = line.trim().split("\\s+");
                    loss += Double.parseDouble(parts[1]);
                }

            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Loss file doesn't exist in HDFS.");
        }

        return loss;
    }


    public static Double find_cost (String input_file, String  output_file) throws Exception{
        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "KMeans Clustering");

        job.setJarByClass(KMeans_ll.class);
        
        // Config Map phase
        job.setMapperClass(CostMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);	

        // Config Reduce phase.
        job.setReducerClass(CostReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(DoubleWritable.class);
        
        // If output path has already existed, delete it before a job.
        Path output_path = new Path(output_file);
        MultipleOutputs.addNamedOutput(job, "distance", TextOutputFormat.class, IntWritable.class, DoubleWritable.class);
        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }
	
        FileInputFormat.setInputPaths(job, new Path(input_file));
        FileOutputFormat.setOutputPath(job, output_path);

        // job.waitForCompletion(true);

        if (job.waitForCompletion(true)) {
            // If job completed, rename the output file. 
            rename_output_files(output_file);

            Double distance = calculate_loss(output_file);

            System.out.println("LOSS: " + distance);

            return distance;

        } else {
            System.out.println("MAP REDUCE JOB FAIL");
            return 0.0;
        }
    }

    public static ArrayList<String> get_centroid (String file_path) throws IOException
    {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path("hdfs://localhost:9000"+ file_path + "/probability.txt");
        ArrayList<String> new_centroid = new ArrayList<>();

        if (fs.exists(loss_file_path)) {

            try (FSDataInputStream input_stream = fs.open(loss_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    String[] doc_term = parts[0].split("\\|");
                    new_centroid.add(doc_term[1]);
                }

            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Loss file doesn't exist in HDFS.");
        }

        return new_centroid;
    }

    public static ArrayList<String> find_probability (Double loss, Double l,  String input_file, String  output_file) throws Exception{
        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);
        conf.set("loss", loss.toString());
        conf.set("l", l.toString());
        ArrayList<String> new_centroid = new ArrayList<>();

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Find probabilitity");

        job.setJarByClass(KMeans_ll.class);
        
        // Config Map phase
        job.setMapperClass(ProbabilityMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);	

        // Config Reduce phase.
        job.setReducerClass(ProbabilityReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
    
        Path output_path = new Path(output_file);
        MultipleOutputs.addNamedOutput(job, "probability", TextOutputFormat.class, Text.class, DoubleWritable.class);

        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }
        FileInputFormat.setInputPaths(job, new Path(input_file));
        FileOutputFormat.setOutputPath(job, output_path);

        if (job.waitForCompletion(true)) {
            // If job completed, rename the output file. 
            rename_output_files(output_file);
            new_centroid = get_centroid(output_file);

        } else {
            System.out.println("MAP REDUCE JOB FAIL");
        }
        return new_centroid;
        
    }

    public static class WeightMapper extends Mapper<Object, Text, Text, IntWritable>{
        public static ArrayList<Map<Integer, Double>> mapper_centroids = new  ArrayList<>();
        public static ArrayList<String> centroids_str_list = new ArrayList<>();
        public final static IntWritable one = new IntWritable(1);

        protected void setup(Context context) throws IOException, InterruptedException {
            // Load centroids from context
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
                    term_tfidf.put(term_id, tfidf);
                }
                mapper_centroids.add(term_tfidf);
                centroids_str_list.add(centroid);
            }            
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] doc_term_tfidf = value.toString().split("\\|");
            Map<Integer, Double> term_tfidf = parse_term_tfidf(doc_term_tfidf[1]);

            // double min_distance = Double.MAX_VALUE;
            // int cluster = -1;
           
            double max_similarity = -1;
            int cluster = -1;
           
           for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double similarity =   KMeans_ll.cosine_similarity(term_tfidf, centroid);

                if (similarity > max_similarity) {
                    cluster = cluster_id;
                    max_similarity = similarity;
                }
            }
        //    for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

        //         Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

        //         double distance =   KMeans_ll.sum_squares(term_tfidf, centroid);
        //         if (distance < min_distance){
        //             min_distance = distance;
        //             cluster = cluster_id;
        //         }
        //     }
            // context.write(new Text(centroids_str_list.get(cluster)), one);

            if (cluster != -1) {
                context.write( new Text(centroids_str_list.get(cluster)), new IntWritable(1));
            }
        }
    }

    public static class WeightReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private MultipleOutputs<Text, IntWritable > mos;
        private IntWritable total_weight = new IntWritable();
      
        public void setup(Context context) throws IOException, InterruptedException {
         
            mos = new MultipleOutputs<>(context);
        }
        
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }

            // total_weight.set(sum);
            mos.write("weight", key, new IntWritable(sum), "weight.txt");

        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static ArrayList<String> find_weight (String input_file, String  output_file) throws Exception{
        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);
        
        String converted_input_file = input_file + "/tfidf.txt";
        ArrayList<String> centroid_weight = new ArrayList<>();

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Find weight");

        job.setJarByClass(KMeans_ll.class);
        
        // Config Map phase
        job.setMapperClass(WeightMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);	

        // Config Reduce phase.
        job.setReducerClass(WeightReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
    
        Path output_path = new Path(output_file);
        MultipleOutputs.addNamedOutput(job, "weight", TextOutputFormat.class, Text.class, IntWritable.class);

        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }
        FileInputFormat.setInputPaths(job, new Path(converted_input_file));
        FileOutputFormat.setOutputPath(job, output_path);

        if (job.waitForCompletion(true)) {
            // If job completed, rename the output file. 
            rename_output_files(output_file);
            centroid_weight = get_weight(output_file);
            move_weight(output_file, input_file);

        } else {
            System.out.println("MAP REDUCE JOB FAIL");
        }
        return centroid_weight;
        
    }

    public static ArrayList<String> get_weight(String file_path) throws IOException
    {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path weight_file_path = new Path("hdfs://localhost:9000"+ file_path + "/weight.txt");
        ArrayList<String> centroid_weight = new ArrayList<>();

        if (fs.exists(weight_file_path)) {

            try (FSDataInputStream input_stream = fs.open(weight_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    centroid_weight.add(line);
                }

            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Loss file doesn't exist in HDFS.");
        }

        return centroid_weight;
    }

    public static void move_weight(String src_file, String des_file) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
    
        Path src_file_path = new Path(src_file + "/weight.txt");
        Path des_file_path = new Path(des_file + "/weight.txt");
        ArrayList<String> lines = new ArrayList<>();
    
        if (fs.exists(src_file_path)) {


            if (!fs.exists(des_file_path)){
                fs.createNewFile(des_file_path);
            }

            try (FSDataInputStream input_stream = fs.open(src_file_path);
                FSDataOutputStream output_stream = fs.create(des_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    // lines.add(line);
                    output_stream.writeBytes(line + "\n");
                }

                
            System.out.println("Weight file copied successfully to input directory.");
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }

       
            
            // try (FSDataOutputStream output_stream = fs.append(des_file_path)){

            //     for (int i = 0; i < lines.size(); i++)
            //     {
                   
            //         String line = lines.get(i) + "\n";
            //         output_stream.write(line.getBytes()); 
            //     }
            // }catch (IOException e) {
            //     System.out.println("Exception: " + e.getMessage());
            // }
            
        } else {
            System.out.println("Weight file doesn't exist in output directory.");
        }

        
    }


    public static class ReclusterMapper extends Mapper<Object, Text, IntWritable, Text>{
        public static ArrayList<Double> mapper_centroids = new  ArrayList<>();

        public void setup(Context context) throws IOException, InterruptedException {
            // Load centroids from context
            Configuration conf = context.getConfiguration();
            String centroids_string = conf.get("centroids");
            String[] centroids_list = centroids_string.split(";");

            for (String centroid : centroids_list){
                Double cen = Double.parseDouble(centroid.trim());
                mapper_centroids.add(cen);
            }            
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] centroid_weight = value.toString().split("\\s+");
            Double weight = Double.parseDouble(centroid_weight[1]);
           
            double min_distance  = Double.MAX_VALUE;
            int cluster = -1;
           
           for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

                Double centroid = mapper_centroids.get(cluster_id);

                double distance = Math.abs(weight - centroid);

                if (distance < min_distance) {
                    cluster = cluster_id;
                    min_distance = distance;
                }
            }
    
            if (cluster != -1) {
                context.write( new IntWritable(cluster), new Text(value));
            }
        }
    }

    public static class ReclusterReducer extends Reducer<IntWritable, Text, Text, IntWritable> {

        private MultipleOutputs <Text, IntWritable > mos;
   
        public void setup(Context context) throws IOException, InterruptedException {
            mos = new MultipleOutputs<>(context);
        }
        
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        
            Double sum_weight = 0.0;
            int num_values = 0;
            for (Text value : values) {
                String[] centroid_weight = value.toString().split("\\s+");
                Double weight = Double.parseDouble(centroid_weight[1]);
                sum_weight += weight;

                mos.write("cluster", value, key, "classes.txt");
                num_values++;
            }

            Double new_weight = sum_weight / num_values;
            String centroid = new_weight.toString();
            mos.write("centroid", new Text(centroid), key, "clusters.txt");
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static String random_k_weight(int k, ArrayList<String> centroid_weight){

        ArrayList<Double> centroid = new ArrayList<>();
        StringBuilder random_weights = new StringBuilder();

        for (int i = 0; i < centroid_weight.size(); i++){
            String[] parts = centroid_weight.get(i).split("\\s+");
            centroid.add(Double.parseDouble(parts[1]));
        }

        Collections.shuffle(centroid);

        for (int i = 0; i < k; i++)
        {
            random_weights.append(centroid.get(i)).append(";");   
        }
        random_weights.deleteCharAt(random_weights.length() - 1);

        return random_weights.toString();
    }

    private static String read_centroid_2(String centroid_path, int k) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path centroid_file_path = new Path("hdfs://localhost:9000" + centroid_path);
        StringBuilder centroids = new StringBuilder();

        if (fs.exists(centroid_file_path)) {

            try (FSDataInputStream input_stream = fs.open(centroid_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\\s+");
                    centroids.append(parts[0]).append(";");
                }
                centroids.deleteCharAt(centroids.length() - 1);

        
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Centroid file doesn't exist in HDFS.");
        }

        return centroids.toString();
    }

    public static void re_cluster (int k, String centroid_weight, String input_file, String  output_file) throws Exception{
        Configuration conf = new Configuration();
        conf.set("centroids", centroid_weight);
        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Recluster");

        job.setJarByClass(KMeans_ll.class);
        
        // Config Map phase
        job.setMapperClass(ReclusterMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);	

        // Config Reduce phase.
        job.setReducerClass(ReclusterReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
    
        Path output_path = new Path(output_file);
        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }

        FileInputFormat.setInputPaths(job, new Path(input_file + "/weight.txt"));
        FileOutputFormat.setOutputPath(job, output_path);
        MultipleOutputs.addNamedOutput(job, "centroid", TextOutputFormat.class, Text.class, IntWritable.class);
        MultipleOutputs.addNamedOutput(job, "cluster", TextOutputFormat.class, Text.class, IntWritable.class);
        
        if (job.waitForCompletion(true)) {
            rename_output_files(output_file);
        } else {
            System.out.println("MAP REDUCE JOB FAIL");
        }        
    }

    public static int has_converged(String old_centroids, String new_centroids, double threshold) {
        // Check if centroids have converged based on a threshold
        int result = 1;
        ArrayList<Double> old_cen = new ArrayList<>();
        ArrayList<Double> new_cen  = new ArrayList<>();

        String[] old_centroids_list = old_centroids.split(";");
        String[] new_centroids_list = new_centroids.split(";");

        if (old_centroids_list.length != new_centroids_list.length){
            return -1;
        }

        for (String centroid : old_centroids_list){

            Double cen = Double.parseDouble(centroid.trim());
            old_cen.add(cen);
        }            

        for (String centroid : new_centroids_list){

            Double cen = Double.parseDouble(centroid.trim());
            new_cen.add(cen);
        }

        
        for (int i = 0; i < old_cen.size(); i++) {

            Double old_centroid = old_cen.get(i);
            Double new_centroid = new_cen.get(i);
            double sum = Math.pow(old_centroid - new_centroid, 2);
            double distance = Math.sqrt(sum);

            System.out.println("Distance: " + distance);
            if (distance > threshold) {
                result = 0;
            }
        }
        if (result == 1){
            System.out.println("Centroids have converged.");
        }else{
            System.out.println("Centroids are still changing.");
        }
        return result;
    }

    private static String get_final_centroid(String centroid_path, int k) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path centroid_file_path = new Path("hdfs://localhost:9000"+ centroid_path);
        ArrayList<Map<Integer, Double>> centroid_maps = new ArrayList<>(k);
        
        StringBuilder final_centroids = new StringBuilder();

        for (int i = 0; i < k; i++) {
            centroid_maps.add(new HashMap<>());
        }

        if (fs.exists(centroid_file_path)) {

            try (FSDataInputStream input_stream = fs.open(centroid_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\\s+");
                    Integer cluster  = Integer.parseInt(parts[2]);
                    String[] term_tfidf_pairs = parts[0].split(",");

                    for (String pair : term_tfidf_pairs) {
                        String[] term_tfidf = pair.split(":");
                        int term = Integer.parseInt(term_tfidf[0]);
                        double tfidf = Double.parseDouble(term_tfidf[1]);
    
                        // Thêm giá trị tfidf vào bản đồ của cụm tương ứng
                        centroid_maps.get(cluster).merge(term, tfidf, Double::sum);
                    }
                    
                }


                for (Map<Integer, Double> centroid_map : centroid_maps) {
                    int size = centroid_map.size();
                    for (Map.Entry<Integer, Double> entry : centroid_map.entrySet()) {
                        entry.setValue(entry.getValue() / size);
                    }
                }

                for (Map<Integer, Double> centroid_map : centroid_maps) {
                    for (Map.Entry<Integer, Double> entry : centroid_map.entrySet()) {
                        final_centroids.append(entry.getKey()).append(":").append(entry.getValue()).append(",");
                    }
                    final_centroids.deleteCharAt(final_centroids.length() - 1);
                    final_centroids.append(";");
                }

                final_centroids.deleteCharAt(final_centroids.length() - 1);

        
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("File doesn't exist in HDFS.");
        }
        return final_centroids.toString();
        
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
        String converted_input_file = input_file + "/tfidf.txt";
        centroids.addAll(init_random_centroid(1,converted_input_file));
        Double loss = find_cost(converted_input_file, output_file);

        System.out.println("LOSSSSSS: "+ loss);
        System.out.println((int)Math.ceil(Math.log(loss)));

        for (int i = 0; i < 3; i++){
            ArrayList <String> new_centroid = find_probability(loss, 2.5, converted_input_file, output_file);
            System.out.println("NEW CENTROID:");
            for (int j = 0; j < new_centroid.size(); j++){
                System.out.println(new_centroid.get(j).substring(0,10));
                System.out.println("________________________");
            }
            centroids.addAll(new_centroid);



            System.out.println("CURRENT CENTROID:");
            ArrayList<String> allCentroids = new ArrayList<>(centroids);

            for (int j = 0; j < allCentroids.size(); j++){
                System.out.println(allCentroids.get(j));
                System.out.println("________________________");
            }


        }

        ArrayList<String> centroid_weight = find_weight(input_file, output_file);

        for(int i = 0; i < centroid_weight.size(); i++){
            System.out.println(centroid_weight.get(i));
            System.out.println("++++++++");
        }

        int iteration = 0;
        
        String centroids = "";
		String old_centroids = "";
        String new_centroids = "";


        System.out.println("**************RECLUSTER***************");
        while (true) {

            System.out.println("**************** ITERATION: " + iteration+"*****************");
            if (iteration == 0){
                centroids = random_k_weight(k, centroid_weight);
            }     
            
            if (iteration > 0){
                old_centroids = new String(centroids);
            }
            
            try{
                System.out.println(centroids);  
                re_cluster(k, centroids, input_file, output_file);
                centroids = read_centroid_2(output_file + "/clusters.txt", k);
                
            }catch (Exception e){
                System.out.println(e);
                System.out.println("ERROR");
            }

            if (iteration > 0){
                new_centroids = new String(centroids);

                if (has_converged(old_centroids, new_centroids,0.5) == 1) {
                    break;
                }
            }
        

        iteration ++;
        }

        String final_centroids = get_final_centroid(output_file + "/classes.txt", k);

        System.out.println(final_centroids);
    }

}
