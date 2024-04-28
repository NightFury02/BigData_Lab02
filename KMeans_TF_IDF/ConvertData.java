import java.io.IOException;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;

import java.util.Map;
import java.util.Random;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
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

public class ConvertData {

    public static class ConvertMapper extends Mapper<Object, Text, IntWritable, Text>{

        private int num_of_term;
        private int num_of_doc;
        private boolean is_matrix_description = true;

        public void map(Object key,  Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();

            if (is_matrix_description) {
            
                String[] parts = line.split(" ");
                if (parts[0].equals("%%MatrixMarket")) {
                    continue;
                } else {
                    // Extract num_term_id, num_doc_id
                    String[] matrixInfo = line.split(" ");
                    num_of_term = Integer.parseInt(matrixInfo[0]);
                    num_of_doc = Integer.parseInt(matrixInfo[1]);
                    is_matrix_description = false;
                }
            }else{
                String[] parts = line.split(" ");
                int term_id = Integer.parseInt(parts[0]);
                int col_id = Integer.parseInt(parts[1]);
                double tf_idf = Double.parseDouble(parts[2]);

                String term_tfidf = term_id + ":" + tf_idf;
                contex.write()

            }
            

        }
    }

    public static class IntSumReducer extends Reducer <IntWritable, Text, Text, Text> {
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            
        }
    }
    
}
