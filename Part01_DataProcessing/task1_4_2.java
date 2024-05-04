import java.io.IOException;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashSet;
import java.util.Set;
import java.util.List;
import java.util.PriorityQueue;
import java.util.HashMap;
import java.util.Collections;
import java.util.Arrays;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.TreeMap;
import java.lang.Math;
import javax.naming.Context;
import java.util.Map;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapred.nativetask.buffer.DataOutputStream;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

public class task1_4_2 {

    public static class KeyMapper
            extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                // Get doc information
                String skey = itr.nextToken();

                // Get term information
                String svalue = itr.nextToken();

                // Discard tf
                itr.nextToken();
                context.write(new Text(skey), new Text(svalue));
            }
        }
    }

    public static class KeyReducer
            extends Reducer<Text, Text, Text, Text> {

        private TreeMap<String, Integer> idft;
        private Integer doc_cnt;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            // Initialize the priority queue with a capacity of 10 (top 10 elements)
            idft = new TreeMap<String, Integer>();
            doc_cnt = 0;
        }

        public void reduce(Text key, Iterable<Text> values,
                Context context) throws IOException, InterruptedException {
            
            // Count docs
            doc_cnt++;

            // Calculate word frequency per doc
            for (Text val : values) {
                String term = val.toString();

                // add term and number of docs include term
                if (idft.containsKey(term)) {
                    idft.put(term, idft.get(term) + 1);
                } else {
                    idft.put(term, 1);
                }
                //Returns: term, num of doc presence
                // calculate doc's total term
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Calculate IDF
            for (Map.Entry<String, Integer> entry : idft.entrySet()) {
                Double idf = Math.log(((double) doc_cnt / entry.getValue()));
                context.write(new Text(entry.getKey()), new Text(String.format("%.4f", idf)));
            }

        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        // Disable speculative execution for map tasks
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);
        conf.setBoolean("mapreduce.job.ubertask.enable", true);

        Job job = Job.getInstance(conf, "cal idf");
        job.setJarByClass(task1_4_2.class);
        job.setMapperClass(KeyMapper.class);
        job.setReducerClass(KeyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.getConfiguration().set("mapreduce.output.basename", "IDF.mtx");
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
