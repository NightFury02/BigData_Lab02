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

public class task1_4_3 {

    public static class KeyMapper
            extends Mapper<Object, Text, Text, Text> {

        private BufferedReader fis;
        private TreeMap<String, Double> idf;

        private void readStopWordFile(Path stopWordFile) {
            try {
                fis = new BufferedReader(new FileReader(stopWordFile.toString()));
                String stopWord = null;
                while ((stopWord = fis.readLine()) != null) {
                    StringTokenizer itr = new StringTokenizer(stopWord.toString());
                    idf.put(itr.nextToken(), Double.parseDouble(itr.nextToken()));
                }
            } catch (IOException ioe) {
                System.err.println("Exception while reading stop word file '"
                        + stopWordFile + "' : " + ioe.toString());
            }
        }

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            // Initialize the priority queue with a capacity of 10 (top 10 elements)
            idf = new TreeMap<String, Double>();

            try {

                Path[] stopWordFiles = new Path[0];
                stopWordFiles = context.getLocalCacheFiles();
                if (stopWordFiles != null && stopWordFiles.length > 0) {
                    for (Path stopWordFile : stopWordFiles) {
                        readStopWordFile(stopWordFile);
                    }
                }
            } catch (IOException e) {
                System.err.println("Exception reading stop word file: " + e);
            }
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String skey = itr.nextToken();
                // String svalue = itr.nextToken();

                String term = skey.split("_")[1];
                Double dvalue = Double.parseDouble(itr.nextToken()) * idf.get(term);
                context.write(new Text(skey), new Text(String.format("%.4f", dvalue)));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        // Disable speculative execution for map tasks
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);
        conf.setBoolean("mapreduce.job.ubertask.enable", true);

        Job job = Job.getInstance(conf, "cal tfidf");
        job.setJarByClass(task1_4_3.class);
        job.setMapperClass(KeyMapper.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.getConfiguration().set("mapreduce.output.basename", "TFIDF.mtx");
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        job.addCacheFile(new Path(args[1]).toUri());
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
