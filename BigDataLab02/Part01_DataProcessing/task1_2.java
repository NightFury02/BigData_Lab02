
import java.io.IOException;
import java.util.StringTokenizer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashSet;
import java.util.Set;
import java.util.List;
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

public class task1_2 {

  public static class KeyMapper
      extends Mapper<Object, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text key_word = new Text();
    private IntWritable value_int = new IntWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

      String skey = "";
      int ivalue = 0;
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {

        skey = itr.nextToken() + "	" + itr.nextToken();
        if (!itr.hasMoreTokens()) {
          break;
        }
        ivalue = Integer.parseInt(itr.nextToken());

        if (ivalue < 3) {
          continue;
        }

        key_word.set(skey);
        value_int.set(ivalue);

        context.write(key_word, value_int);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "filter word by frequency");
    job.setJarByClass(task1_2.class);
    job.setMapperClass(KeyMapper.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    job.getConfiguration().set("mapreduce.output.basename", "task_1_2.mtx");
    FileInputFormat.setInputDirRecursive(job, true);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
