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

public class task1_5_2 {

    public static class KeyMapper
            extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String skey = itr.nextToken();
                String svalue = itr.nextToken();

                String[] cat_doc = skey.split(":");
                // String[] doc_term = cat_doc[1].split("");

                context.write(new Text(cat_doc[0]), new Text(cat_doc[1] + "\t" + svalue));
            }
        }
    }

    public static class KeyReducer
            extends Reducer<Text, Text, Text, Text> {

        private TreeMap<Double, String> tfidf_top;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            tfidf_top = new TreeMap<Double, String>();
            // Initialize the priority queue with a capacity of 10 (top 10 elements)
        }

        public void reduce(Text key, Iterable<Text> values,
                Context context) throws IOException, InterruptedException {

            Double mean_per_category = 0.0;
            String skey = key.toString();
            for (Text val : values) {
                String[] term_freq = val.toString().split("\t");
                Double dval = Double.parseDouble(term_freq[1]);

                tfidf_top.put(dval, term_freq[0]);
                if (tfidf_top.size() > 5) {
                    tfidf_top.remove(tfidf_top.firstKey());
                }
            }
            
            String svalue = "";
            for (Map.Entry<Double, String> entry : tfidf_top.entrySet()) {
                double count = entry.getKey();
                String term = entry.getValue();

                svalue += term + ":" + String.valueOf(count) + ", ";
            }

            svalue = svalue.substring(0, svalue.length() - 2);

            tfidf_top.clear();

            context.write(new Text(skey + ":"), new Text(svalue));

        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);
        conf.setBoolean("mapreduce.job.ubertask.enable", true);

        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(task1_5_2.class);
        job.setMapperClass(KeyMapper.class);
        job.setReducerClass(KeyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.getConfiguration().set("mapreduce.output.basename", "cTFIDF_5.mtx");
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
