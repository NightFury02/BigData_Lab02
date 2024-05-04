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

public class task2_3 {
    public static String fs_default_path = "hdfs://localhost:9000";
    public static Set<String> centroids = new HashSet<>();
    public static ArrayList<Double> iter_loss = new ArrayList<>();
    public static ArrayList<String> top_terms = new ArrayList<>();


    public static ArrayList<String> init_random_centroid(int k, String input_path) throws IOException {
        ArrayList<String> random_centroids = new ArrayList<>();
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path input_file_path = new Path(fs_default_path + input_path);

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

                for (int i = 0; i < k; i++) {
                    line = lines.get(i);
                    String[] parts = line.split("\\s+");
                    random_centroids.add(parts[1].trim());
                }
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }

            for (int i = 0; i < random_centroids.size(); i++) {
                System.out.println(random_centroids.get(i));
            }
        }
        return random_centroids;
    }

    public static String centroids_to_string() {
        StringBuilder centroids_string = new StringBuilder();

        for (String centroid : centroids) {
            centroids_string.append(centroid).append(";");
        }

        centroids_string.deleteCharAt(centroids_string.length() - 1);
        return centroids_string.toString();
    }

    public static Map<Integer, Double> parse_term_tfidf(String line) {

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

    public static void rename_output_files(String output_dir) throws IOException {
        FileSystem hdfs = FileSystem.get(new Configuration());
        Path output_path = new Path(output_dir);
        FileStatus fs[] = hdfs.listStatus(output_path);

        if (fs != null) {
            for (FileStatus file : fs) {
                if (!file.isDir()) {
                    String file_name = file.getPath().getName();
                    int index = file_name.indexOf("-r-00000");

                    if (index != -1) {
                        String new_file_name = file_name.substring(0, index);
                        Path new_path = new Path(fs_default_path + output_dir + "/" + new_file_name);

                        hdfs.rename(file.getPath(), new_path);
                    }
                }
            }
        }
    }

    public static double sum_squares(Map<Integer, Double> term_tfidf, Map<Integer, Double> centroid) {
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

    public static double cosine_similarity(Map<Integer, Double> doc1, Map<Integer, Double> doc2) {
        double dot_product = 0.0;

        for (Map.Entry<Integer, Double> entry : doc1.entrySet()) {
            int term_id = entry.getKey();

            if (doc2.containsKey(term_id)) {
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
        return Math.round(dot_product * 1000.0 / (doc1_l2_length * doc2_l2_length)) / 1000.0;
    }

    public static ArrayList<Map<Integer, Double>> parse_centroid(String centroids_string) {
        ArrayList<Map<Integer, Double>> parsed_centroids = new ArrayList<>();
        String[] centroids_list = centroids_string.split(";");

        for (String centroid : centroids_list) {

            Map<Integer, Double> term_tfidf = new HashMap<>();
            String[] term_tfidf_pairs = centroid.split(",");

            for (String pair : term_tfidf_pairs) {

                String[] parts = pair.split(":");
                int term_id = Integer.parseInt(parts[0]);
                double tfidf = Double.parseDouble(parts[1]);
                term_tfidf.put(term_id, tfidf);
            }
            parsed_centroids.add(term_tfidf);
        }
        return parsed_centroids;
    }

    public static class CostMapper extends Mapper<Object, Text, IntWritable, DoubleWritable> {

        public static ArrayList<Map<Integer, Double>> mapper_centroids;

        public void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            String centroids_string = conf.get("centroids");
            mapper_centroids = new ArrayList<>(task2_3.parse_centroid(centroids_string));
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] doc_term_tfidf = value.toString().split("\\s+");
           
            Map<Integer, Double> term_tfidf = parse_term_tfidf(doc_term_tfidf[1]);
            double min_distance = Double.MAX_VALUE;
            int cluster = -1;

            for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id++) {
                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);
                double distance = task2_3.sum_squares(term_tfidf, centroid);

                if (distance < min_distance) {
                    min_distance = distance;
                    cluster = cluster_id;
                }
            }
            context.write(new IntWritable(cluster), new DoubleWritable(min_distance));
        }
    }

    public static class CostReducer extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
        private MultipleOutputs<IntWritable, DoubleWritable> mos;

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

    public static class ProbabilityMapper extends Mapper<Object, Text, Text, DoubleWritable> {

        public static ArrayList<Map<Integer, Double>> mapper_centroids;
        private Double total_loss;
        private Double over_sampling;

        public void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();

            String centroids_string = conf.get("centroids");
            mapper_centroids = new ArrayList<>(parse_centroid(centroids_string));

            String loss_str = conf.get("loss");
            total_loss = Double.parseDouble(loss_str);

            String l = conf.get("l");
            over_sampling = Double.parseDouble(l);
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] doc_term_tfidf = value.toString().split("\\s+");
            Map<Integer, Double> term_tfidf = parse_term_tfidf(doc_term_tfidf[1]);
            double min_distance = Double.MAX_VALUE;
            int cluster = -1;

            for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double distance = task2_3.sum_squares(term_tfidf, centroid);
                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
            Double probability = over_sampling * min_distance / total_loss;

            context.write(value, new DoubleWritable(probability));
        }
    }

    public static class ProbabilityReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        private Map<Text, Double> candidate_centroids = new HashMap<>();
        private Double over_sampling;
        private MultipleOutputs<Text, DoubleWritable> mos;

        public void setup(Context context) throws IOException, InterruptedException {

            mos = new MultipleOutputs<>(context);
            Configuration conf = context.getConfiguration();
            String l = conf.get("l");
            over_sampling = Double.parseDouble(l);
        }

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double max_probability = Double.MIN_VALUE;
            for (DoubleWritable value : values) {
                if (value.get() > max_probability) {
                    max_probability = value.get();
                }
            }

            candidate_centroids.put(new Text(key), max_probability);
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            List<Map.Entry<Text, Double>> sorted_entries = new ArrayList<>(candidate_centroids.entrySet());
            sorted_entries.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
            int count = 1;

            for (Map.Entry<Text, Double> entry : sorted_entries) {
                if (count > over_sampling) {
                    break;
                }
                mos.write("probability", entry.getKey(), new DoubleWritable(entry.getValue()), "probability.txt");
                count++;
            }
            mos.close();
        }
    }

    public static Double calculate_cost(String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path cost_file_path = new Path(fs_default_path + file_path + "/distances.txt");
        Double cost = 0.0;

        if (fs.exists(cost_file_path)) {

            try (FSDataInputStream input_stream = fs.open(cost_file_path);
                    InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                    BufferedReader reader = new BufferedReader(input_stream_reader)) {
                String line;

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    cost += Double.parseDouble(parts[1]);
                }
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Loss file doesn't exist in HDFS.");
        }
        return cost;
    }

    public static Double find_cost(String input_file, String output_file) throws Exception {
        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Find cost");

        job.setJarByClass(task2_3.class);

        // Config Map phase
        job.setMapperClass(CostMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);

        // Config Reduce phase.
        job.setReducerClass(CostReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(DoubleWritable.class);

        Path output_path = new Path(output_file);
        MultipleOutputs.addNamedOutput(job, "distance", TextOutputFormat.class, IntWritable.class,
                DoubleWritable.class);
        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }

        FileInputFormat.setInputPaths(job, new Path(input_file));
        FileOutputFormat.setOutputPath(job, output_path);

        if (job.waitForCompletion(true)) {
            // If job completed, rename the output file.
            rename_output_files(output_file);

            Double distance = calculate_cost(output_file);

            System.out.println("TOTAL COST: " + distance);

            return distance;

        } else {
            System.out.println("MAP REDUCE JOB FAIL");
            return 0.0;
        }
    }

    public static ArrayList<String> get_candidate_centroid(String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path(fs_default_path + file_path + "/probability.txt");
        ArrayList<String> new_centroid = new ArrayList<>();

        if (fs.exists(loss_file_path)) {

            try (FSDataInputStream input_stream = fs.open(loss_file_path);
                    InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                    BufferedReader reader = new BufferedReader(input_stream_reader)) {
                String line;

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    new_centroid.add(parts[1]);
                }

            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Probability file doesn't exist in HDFS.");
        }
        return new_centroid;
    }

    public static ArrayList<String> find_probability(Double loss, Double l, String input_file, String output_file)
            throws Exception {
        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);
        conf.set("loss", loss.toString());
        conf.set("l", l.toString());
        ArrayList<String> new_centroid = new ArrayList<>();

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Find probabilitity");

        job.setJarByClass(task2_3.class);

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

            rename_output_files(output_file);
            new_centroid = get_candidate_centroid(output_file);

        } else {
            System.out.println("MAP REDUCE JOB FAIL");
        }
        return new_centroid;
    }

    public static class WeightMapper extends Mapper<Object, Text, Text, IntWritable> {
        public static ArrayList<Map<Integer, Double>> mapper_centroids = new ArrayList<>();
        public static ArrayList<String> centroids_str_list = new ArrayList<>();
        public final static IntWritable one = new IntWritable(1);

        protected void setup(Context context) throws IOException, InterruptedException {
            // Load centroids from context
            Configuration conf = context.getConfiguration();
            String centroids_string = conf.get("centroids");
            String[] centroids_list = centroids_string.split(";");

            for (String centroid : centroids_list) {

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
            String[] doc_term_tfidf = value.toString().split("\\s+");
            Map<Integer, Double> term_tfidf = parse_term_tfidf(doc_term_tfidf[1]);
            double max_similarity = -1;
            int cluster = -1;

            for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double similarity = task2_3.cosine_similarity(term_tfidf, centroid);

                if (similarity > max_similarity) {
                    cluster = cluster_id;
                    max_similarity = similarity;
                }
            }

            if (cluster != -1) {
                context.write(new Text(centroids_str_list.get(cluster)), new IntWritable(1));
            }
        }
    }

    public static class WeightReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private MultipleOutputs<Text, IntWritable> mos;
        private IntWritable total_weight = new IntWritable();

        public void setup(Context context) throws IOException, InterruptedException {

            mos = new MultipleOutputs<>(context);
        }

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }

            mos.write("weight", key, new IntWritable(sum), "weight.txt");

        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static ArrayList<String> find_weight(String input_file, String output_file) throws Exception {
        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);

        String converted_input_file = input_file + "/tfidf.txt";
        ArrayList<String> centroid_weight = new ArrayList<>();

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Find weight");

        job.setJarByClass(task2_3.class);

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

    public static ArrayList<String> get_weight(String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path weight_file_path = new Path(fs_default_path + file_path + "/weight.txt");
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

            if (!fs.exists(des_file_path)) {
                fs.createNewFile(des_file_path);
            }

            try (FSDataInputStream input_stream = fs.open(src_file_path);
                    FSDataOutputStream output_stream = fs.create(des_file_path);
                    InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                    BufferedReader reader = new BufferedReader(input_stream_reader)) {

                String line;

                while ((line = reader.readLine()) != null) {
                    output_stream.writeBytes(line + "\n");
                }

                System.out.println("Weight file copied successfully to input directory.");
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }

        } else {
            System.out.println("Weight file doesn't exist in output directory.");
        }

    }

    public static class ReclusterMapper extends Mapper<Object, Text, IntWritable, Text> {
        public static ArrayList<Map<Integer, Double>> mapper_centroids = new ArrayList<>();

        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();

            String centroids_string = conf.get("centroids");
            mapper_centroids = task2_3.parse_centroid(centroids_string);
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] centroid_weight = value.toString().split("\\s+");

            Map<Integer, Double> term_tfidf = task2_3.parse_term_tfidf(centroid_weight[0]);
            Double weight = Double.parseDouble(centroid_weight[1]);

            double max_similarity = -1;
            int cluster = -1;

            for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);
                double similarity = task2_3.cosine_similarity(term_tfidf, centroid);

                if (similarity > max_similarity) {
                    cluster = cluster_id;
                    max_similarity = similarity;
                }
            }

            if (cluster != -1) {
                context.write(new IntWritable(cluster), new Text(value));
            }
        }
    }

    public static class ReclusterReducer extends Reducer<IntWritable, Text, Text, IntWritable> {

        private MultipleOutputs<Text, IntWritable> mos;

        public void setup(Context context) throws IOException, InterruptedException {
            mos = new MultipleOutputs<>(context);
        }

        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            Map<Integer, Double> new_centroid = new HashMap<>();
            Double sum_weight = 0.0;
            int num_values = 0;

            for (Text value : values) {
                String[] centroid_weight = value.toString().split("\\s+");
                Double weight = Double.parseDouble(centroid_weight[1]);
                sum_weight += weight;

                Map<Integer, Double> term_tfidf = task2_3.parse_term_tfidf(centroid_weight[0]);

                for (Map.Entry<Integer, Double> entry : term_tfidf.entrySet()) {
                    int term_id = entry.getKey();
                    double tfidf = entry.getValue();

                    new_centroid.put(term_id, new_centroid.getOrDefault(term_id, 0.0) + weight * tfidf);
                }
                mos.write("cluster", value, key, "classes.txt");
                num_values++;
            }

            StringBuilder new_centroid_builder = new StringBuilder();

            for (Map.Entry<Integer, Double> entry : new_centroid.entrySet()) {
                int term_id = entry.getKey();
                double sum_tfidf = entry.getValue();

                double avg_tfidf = sum_tfidf / (sum_weight);

                new_centroid.put(term_id, avg_tfidf);

                new_centroid_builder.append(term_id).append(":").append(avg_tfidf).append(",");
            }
            new_centroid_builder.deleteCharAt(new_centroid_builder.length() - 1);
            mos.write("centroid", new Text(new_centroid_builder.toString()), key, "clusters.txt");
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static ArrayList<String> random_k_weight(int k, ArrayList<String> centroid_weight) {

        ArrayList<String> centroids_list = new ArrayList<>();
        for (int i = 0; i < centroid_weight.size(); i++) {
            String centroid = centroid_weight.get(i).split("\\s+")[0];
            centroids_list.add(centroid);
        }

        Collections.shuffle(centroids_list);

        ArrayList<String> res = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            res.add(centroids_list.get(i));
        }
        return res;
    }

    private static void read_centroids(String centroid_path, int k) throws IOException {

        System.out.println("*******READ CENTROID FUNCTION**********");

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // Path to centroids file in HDFS
        Path centroid_file_path = new Path(fs_default_path + centroid_path);

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

                System.out.println("CENTROID SIZE: " + centroids.size());
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("Centroid file doesn't exist in HDFS.");
        }
        System.out.println("********FINISH READING CENTROIDS**********");
    }

    public static void re_cluster(int k, String input_file, String output_file) throws Exception {
        Configuration conf = new Configuration();
        String centroid_weight = centroids_to_string();
        conf.set("centroids", centroid_weight);
        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "Recluster");

        job.setJarByClass(task2_3.class);

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

    public static boolean has_converged_vector(ArrayList<String> old_centroids, ArrayList<String> new_centroids,
            double threshold) {
        System.out.println("********CHECKING CONVERGENGE FUNCTION*********");

        boolean result = true;

        for (int i = 0; i < old_centroids.size(); i++) {

            Map<Integer, Double> old_centroid = parse_term_tfidf(old_centroids.get(i));
            Map<Integer, Double> new_centroid = parse_term_tfidf(new_centroids.get(i));

            double similarity = cosine_similarity(old_centroid, new_centroid);
            // double similarity = sum_squares(old_centroid, new_centroid);

            System.out.println("Similarity " + i + ": " + similarity);

            if (similarity < threshold) {
                result = false;
            }
        }

        if (result == true) {
            System.out.println("Centroids have converged.");
        } else {
            System.out.println("Centroids are still changing.");
        }
        return result;
    }

    private static ArrayList<String> get_final_centroid(String centroid_path, int k) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path centroid_file_path = new Path(fs_default_path + centroid_path);
        ArrayList<Map<Integer, Double>> centroid_maps = new ArrayList<>(k);
        ArrayList<String> final_centroids = new ArrayList<>();

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
                    final_centroids.add(parts[0]);

                }

            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("File doesn't exist in HDFS.");
        }

        return final_centroids;
    }

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {

        public static ArrayList<Map<Integer, Double>> mapper_centroids = new ArrayList<>();

        public void setup(Context context) throws IOException, InterruptedException {

            // Get centroids string from Kmeans class to Mapper class
            Configuration conf = context.getConfiguration();
            String centroids_string = conf.get("centroids");
            // mapper_centroids = new ArrayList<>(parse_centroid(centroids_string));
            String[] centroids_list = centroids_string.split(";");

            for (String centroid : centroids_list) {

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
            String[] doc_term_tfidf = value.toString().split("\\s+");
            int doc_id = Integer.parseInt(doc_term_tfidf[0]);

            Map<Integer, Double> term_tfidf = task2_3.parse_term_tfidf(doc_term_tfidf[1]);

            double max_similarity = -1;
            int cluster = -1;

            for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id++) {

                Map<Integer, Double> centroid = mapper_centroids.get(cluster_id);

                double similarity = task2_3.cosine_similarity(term_tfidf, centroid);

                if (similarity > max_similarity) {
                    cluster = cluster_id;
                    max_similarity = similarity;
                }
            }
            context.write(new IntWritable(cluster), value);
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Text, Text, IntWritable> {

        private MultipleOutputs<Text, IntWritable> mos;
        private ArrayList<Map<Integer, Double>> line_values = new ArrayList<>();

        public void setup(Context context) throws IOException, InterruptedException {
            mos = new MultipleOutputs<>(context);
        }

        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Map<Integer, Double> new_centroid = new HashMap<>();
            int num_values = 0;

            for (Text value : values) {
                String[] doc_term_tfidf = value.toString().split("\\s+");

                int doc_id = Integer.parseInt(doc_term_tfidf[0]);
                Map<Integer, Double> term_tfidf = task2_3.parse_term_tfidf(doc_term_tfidf[1]);
                line_values.add(term_tfidf);

                // Find the sum of tfidf in cluster
                for (Map.Entry<Integer, Double> entry : term_tfidf.entrySet()) {
                    int term_id = entry.getKey();
                    double tfidf = entry.getValue();

                    new_centroid.put(term_id, new_centroid.getOrDefault(term_id, 0.0) + tfidf);
                }
                num_values++;
                mos.write("cluster", new Text(Integer.toString(doc_id)), key, "task_2_3.classes");
            }

            List<Map.Entry<Integer, Double>> sorted_entries = new ArrayList<Map.Entry<Integer, Double>>(
                    new_centroid.entrySet());
            sorted_entries.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
            List<Map.Entry<Integer, Double>> top_10_entries = sorted_entries.subList(0,
                    Math.min(10, sorted_entries.size()));

            StringBuilder top_10_terms = new StringBuilder();

            for (Map.Entry<Integer, Double> entry : top_10_entries) {
                int term_id = entry.getKey();
                double tfidf = entry.getValue();
                top_10_terms.append(term_id).append(":").append(tfidf).append(",");
            }
            top_10_terms.deleteCharAt(top_10_terms.length() - 1);
            mos.write("topterms", new Text(top_10_terms.toString()), key, "task_2_3.topterms");

            StringBuilder new_centroid_builder = new StringBuilder();

            for (Map.Entry<Integer, Double> entry : new_centroid.entrySet()) {
                int term_id = entry.getKey();
                double sum_tfidf = entry.getValue();

                double avg_tfidf = sum_tfidf / num_values;

                new_centroid.put(term_id, avg_tfidf);

                new_centroid_builder.append(term_id).append(":").append(avg_tfidf).append(",");
            }
            new_centroid_builder.deleteCharAt(new_centroid_builder.length() - 1);
            mos.write("centroid", new Text(new_centroid_builder.toString()), key, "task_2_3.clusters");

            double sum_squared_distances = 0.0;

            for (int i = 0; i < line_values.size(); i++) {
                Map<Integer, Double> term_tfidf = line_values.get(i);
                sum_squared_distances += task2_3.sum_squares(term_tfidf, new_centroid);
            }

            mos.write("loss", new Text(Double.toString(sum_squared_distances)), key, "cluster.loss");
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    public static void get_top_terms(int iteration, String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path(fs_default_path + file_path + "/task_2_3.topterms");

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

    public static void write_top_terms(int k, String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path output_file_path = new Path(fs_default_path + file_path + "/task_2_3.txt");

        if (!fs.exists(output_file_path)) {
            fs.createNewFile(output_file_path);
        }

        int count = 0;
        int iter = 0;
        try (FSDataOutputStream output_stream = fs.append(output_file_path)) {
            for (int i = 0; i < top_terms.size(); i++) {
                if (count == k) {
                    count = 0;
                }
                if (count == 0) {
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

    public static void calculate_loss(int iteration, String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path loss_file_path = new Path(fs_default_path + file_path + "/cluster.loss");

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

    public static void write_loss(String file_path) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path output_file_path = new Path(fs_default_path + file_path + "/task_2_3.loss");

        if (!fs.exists(output_file_path)) {
            fs.createNewFile(output_file_path);
        }

        try (FSDataOutputStream output_stream = fs.append(output_file_path)) {
            for (int i = 0; i < iter_loss.size(); i++) {
                String output_line = "Iteration: " + i + ", Loss = " + iter_loss.get(i) + "\n";
                output_stream.write(output_line.getBytes());
            }

        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
        }
    }

    public static void run_map_reduce_job(int iteration, int k, String input_file, String output_file)
            throws Exception {

        Configuration conf = new Configuration();
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "KMeans Clustering");

        job.setJarByClass(task2_3.class);

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
        MultipleOutputs.addNamedOutput(job, "centroid", TextOutputFormat.class, Text.class, IntWritable.class);
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
        try {
            run_map_reduce_job(iteration, k, input_file, output_file);
            read_centroids(output_file + "/task_2_3.clusters", k);
            calculate_loss(iteration, output_file);
            get_top_terms(iteration, output_file);

        } catch (Exception e) {
            System.out.println("Exception: " + e.getMessage());
            System.out.println("ERROR");
        }
    }

    public static ArrayList<String> get_k_most_weight_centroid(int k, ArrayList<String> centroids_weights) {
        ArrayList<String> centroids_list = new ArrayList<>();
        Map<String, Double> centroid_weight_map = new HashMap<>();

        for (int i = 0; i < centroids_weights.size(); i++) {
            String[] parts = centroids_weights.get(i).split("\\s+");
            String centroid = parts[0];
            Double weight = Double.parseDouble(parts[1]);

            centroid_weight_map.put(centroid, weight);
        }

        List<Map.Entry<String, Double>> sorted_map = new ArrayList<Map.Entry<String, Double>>(
                centroid_weight_map.entrySet());
        sorted_map.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
        List<Map.Entry<String, Double>> top_k = sorted_map.subList(0, Math.min(k, sorted_map.size()));

        for (Map.Entry<String, Double> entry : top_k) {
            String centroid = entry.getKey();
            centroids_list.add(centroid);
        }
        return centroids_list;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.out.println("Invalid parameters. Use: <input_path> <output_path> <k> <max_iterations> <l>");
            System.exit(1);
        }
        String input_file = args[0];
        String output_file = args[1];
        int k = Integer.parseInt(args[2]);
        int max_iterations = Integer.parseInt(args[3]);
        Double l = Double.parseDouble(args[4]);

        String converted_input_file = input_file + "/tfidf.txt";
        // convert_file(input_file + "/input.mtx", converted_input_file);

        centroids.addAll(init_random_centroid(1, converted_input_file));
        Double loss = find_cost(converted_input_file, output_file);

        System.out.println((int) Math.ceil(Math.log(loss)));

        for (int i = 0; i < (int) Math.ceil(Math.log(loss)); i++) {
            System.out.println("ADD NEW CENTROID ITER: " + i );
            ArrayList<String> new_centroid = find_probability(loss, l, converted_input_file, output_file);
            centroids.addAll(new_centroid);
            loss = find_cost(converted_input_file, output_file);
        }

        ArrayList<String> centroid_weight = find_weight(input_file, output_file);

        for (int i = 0; i < centroid_weight.size(); i++) {
            System.out.println(centroid_weight.get(i));
            System.out.println("++++++++");
        }

        int iteration = 0;
        centroids.clear();
        ArrayList<String> old_centroids = new ArrayList<>();
        ArrayList<String> new_centroids = new ArrayList<>();

        System.out.println("**************RECLUSTER***************");
        while (iteration < max_iterations) {

            System.out.println("**************** ITERATION: " + iteration + "*****************");
            if (iteration == 0) {
                centroids.addAll(random_k_weight(k, centroid_weight));
            }

            old_centroids = new ArrayList<>(centroids);

            try {
                re_cluster(k, input_file, output_file);
                read_centroids(output_file + "/clusters.txt", k);

            } catch (Exception e) {
                System.out.println(e);
                System.out.println("ERROR");
            }

            new_centroids = new ArrayList<>(centroids);

            if (has_converged_vector(old_centroids, new_centroids, 0.8) == true) {
                break;

                // System.out.println("hihi");
            }

            iteration++;
        }

        centroids.clear();
        centroids.addAll(get_final_centroid(output_file + "/clusters.txt", k));
        // centroids.addAll(get_k_most_weight_centroid(k, centroid_weight));

        for (String value : centroids) {
            System.out.println(value);
            System.out.println("+++++++++++++++++++++++++++++++++++++++++++++++++");
        }

        iteration = 0;
        k = Integer.parseInt(args[2]);
        // ArrayList<String> old_centroids = new ArrayList<>();
        // ArrayList<String> new_centroids = new ArrayList<>();

        old_centroids.clear();
        new_centroids.clear();

        String centroids_string = centroids_to_string();

        System.out.println("**************FINAL CLUSTERING***************");
        while (iteration < max_iterations) {
            System.out.println("+++++++++++++++++");
            System.out.println("Iteration: " + iteration);

            old_centroids = new ArrayList<>(centroids);

            run(iteration, k, converted_input_file, output_file);

            new_centroids = new ArrayList<>(centroids);

            if (has_converged_vector(old_centroids, new_centroids, 0.9) == true) {
                break;

            }
            iteration++;
        }

        write_top_terms(k, output_file);
        write_loss(output_file);
    }

}
