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

public class KMeansHelper {

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
    
}
