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

public class task1_4_1 {

    public static class KeyMapper
            extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String skey = itr.nextToken();
                String svalue = itr.nextToken() + "_" + itr.nextToken();
                context.write(new Text(skey), new Text(svalue));
            }
        }
    }

    public static class KeyReducer
            extends Reducer<Text, Text, Text, Text> {

        private Text doc_term = new Text();
        private Text dtfreq = new Text();
        private TreeMap<String, Double> ftd;
        private TreeMap<String, Integer> idft;
        private Integer doc_cnt;

        private TreeMap<String, Double> ft;
        private TreeMap<String, Double> idf;


        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            // Initialize the priority queue with a capacity of 10 (top 10 elements)
            ftd = new TreeMap<String, Double>();
            idft = new TreeMap<String, Integer>();
            ft = new TreeMap<String, Double>();
            doc_cnt = 0;
        }

        public void reduce(Text key, Iterable<Text> values,
                Context context) throws IOException, InterruptedException {
            
            // Count docs
            doc_cnt++;

            // get doc
            String doc = key.toString();

            int count = 0;

            // Calculate word frequency per doc
            for (Text val : values) {
                String[] term_freq = val.toString().split("_");

                // get frequency
                int freq = Integer.parseInt(term_freq[1]);

                // get term
                String term = term_freq[0];

                // get add term frequency
                ftd.put(term, (double) freq);

                // calculate doc's total term
                count += freq;
            }

            String term = "";
            double tfreq = 0;
            for (Map.Entry<String, Double> entry : ftd.entrySet()) {
                // get term
                term = entry.getKey();

                // get freq
                tfreq = (double) entry.getValue() * 1.0 / count;

                doc_term.set(doc + "_" + term);

                dtfreq.set(String.format("%.4f", tfreq));

                // return doc.term frequency
                context.write(doc_term, dtfreq);
            }

            ftd.clear();;
            idft.clear();;
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);
        conf.setBoolean("mapreduce.job.ubertask.enable", true);

        Job job = Job.getInstance(conf, "cal tf");
        job.setJarByClass(task1_4_1.class);
        job.setMapperClass(KeyMapper.class);
        job.setReducerClass(KeyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.getConfiguration().set("mapreduce.output.basename", "TF.mtx");
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
