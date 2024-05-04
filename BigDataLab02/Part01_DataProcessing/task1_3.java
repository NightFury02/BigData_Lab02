
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
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
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

public class task1_3 {

  public static class KeyMapper
      extends Mapper<Object, Text, Text, IntWritable> {

    private Text key_word = new Text();
    private IntWritable value_int = new IntWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      String skey = "";
      int ivalue = 0;
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        itr.nextToken();

        // Get term information
        skey = itr.nextToken();
        ivalue = Integer.parseInt(itr.nextToken());

        key_word.set(skey);
        value_int.set(ivalue);

        context.write(key_word, value_int);
      }
    }
  }

  public static class KeyReducer
      extends Reducer<Text, IntWritable, Text, IntWritable> {

    private TreeMap<Integer, String> tmap2;

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);
      // Initialize the priority queue with a capacity of 10 (top 10 elements)
      tmap2 = new TreeMap<Integer, String>();
    }

    public void reduce(Text key, Iterable<IntWritable> values,
        Context context) throws IOException, InterruptedException {

      int count = 0;
      for (IntWritable val : values) {
        count += val.get();
      }

      // Add count + term to tree map
      tmap2.put(count, key.toString());
      if (tmap2.size() > 10) {
        // Because tree map is always sorted ascendingly
        // If the size of the map is bigger than 10
        // We can remove the top element (the smallest)
        tmap2.remove(tmap2.firstKey());
      }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      for (Map.Entry<Integer, String> entry : tmap2.entrySet()) {
        int count = entry.getKey();
        String term = entry.getValue();

        // And finally, we write the top 10 term and count to output
        context.write(new Text(term), new IntWritable(count));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "top 10 freq words");
    job.setJarByClass(task1_3.class);
    job.setMapperClass(KeyMapper.class);
    job.setReducerClass(KeyReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    job.getConfiguration().set("mapreduce.output.basename", "task_1_3.mtx");
    FileInputFormat.setInputDirRecursive(job, true);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}