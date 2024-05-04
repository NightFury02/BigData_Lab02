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

public class task1_5_1 {

    public static class KeyMapper
            extends Mapper<Object, Text, Text, Text> {
        private Set<String> stopWordList = new HashSet<String>();
        private BufferedReader fis;
        
        private TreeMap<String, String> termid;
        private TreeMap<String, String> docid;

        @SuppressWarnings("deprecation")
        protected void setup(Context context) throws java.io.IOException,
                InterruptedException {

            termid = new TreeMap<String, String>();
            docid = new TreeMap<String, String>();

            try {
                Path[] cachedFiles = context.getLocalCacheFiles();
                Path termidFile = cachedFiles[0];
                Path docidFile = cachedFiles[1];

                if (docidFile != null) {
                    readDocIDFile(docidFile);
                }

                if (termidFile != null) {
                    readTermIDFile(termidFile);
                }

            } catch (IOException e) {
                System.err.println("Exception reading stop word file: " + e);

            }

        }

        void readTermIDFile(Path termidPath) {
            try {
                fis = new BufferedReader(new FileReader(termidPath.toString()));
                String terminfo = null;
                while ((terminfo = fis.readLine()) != null) {
                    String[] term_and_id = terminfo.split("\t");
                    termid.put(term_and_id[1], term_and_id[0]);
                }
            } catch (IOException ioe) {
                System.err.println("Exception while reading termid file '"
                        + termidPath + "' : " + ioe.toString());
            }
        }

        void readDocIDFile(Path docidPath) {
            try {
                fis = new BufferedReader(new FileReader(docidPath.toString()));
                String docinfo = null;
                while ((docinfo = fis.readLine()) != null) {
                    String[] doc_and_id = docinfo.split("\t");
                    docid.put(doc_and_id[1], doc_and_id[0]);
                }
            } catch (IOException ioe) {
                System.err.println("Exception while reading docid file '"
                        + docidPath + "' : " + ioe.toString());
            }
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String skey = itr.nextToken();
                String svalue = itr.nextToken();

                String doc = "";
                String term = "";
                String[] doc_term = skey.split("_");

                if (docid.containsKey(doc_term[0])) {
                    doc = docid.get(doc_term[0]);
                }

                if (termid.containsKey(doc_term[1])) {
                    term = termid.get(doc_term[1]);
                }

                String [] doc_file = doc.split("\\.");
                String dt_info = doc_file[0] + ":" + term;
                context.write(new Text(dt_info), new Text(svalue));

                // if (doc != "" && term != "") {
                //     // String[] doc_file = doc[1].split("\\.");
                //     String dt_info = doc + ":" + term;
                //     context.write(new Text(dt_info), new Text(svalue));
                // }
            }
        }
    }

    public static class KeyReducer
            extends Reducer<Text, Text, Text, Text> {

        private TreeMap<Double, String> tfidf_top;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            // Initialize the priority queue with a capacity of 10 (top 10 elements)
        }

        public void reduce(Text key, Iterable<Text> values,
                Context context) throws IOException, InterruptedException {

            Double mean_per_category = 0.0;
            int cnt = 0;
            for (Text val : values) {
                Double dval = Double.parseDouble(val.toString());
                mean_per_category += dval;
                cnt += 1;
            }
            if (cnt > 0) {
                mean_per_category = (double) mean_per_category * 1.0 / cnt;
            }
            context.write(key, new Text(String.format("%.4f", mean_per_category)));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);
        conf.setBoolean("mapreduce.job.ubertask.enable", true);

        Job job = Job.getInstance(conf, "calculate cTFIDF");
        job.setJarByClass(task1_5_1.class);
        job.setMapperClass(KeyMapper.class);
        job.setReducerClass(KeyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.getConfiguration().set("mapreduce.output.basename", "cTFIDF.mtx");
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        job.addCacheFile(new Path(args[1]).toUri());
        job.addCacheFile(new Path(args[2]).toUri());
        FileOutputFormat.setOutputPath(job, new Path(args[3]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
