
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.TreeMap;

import javax.naming.Context;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashSet;
import java.util.Set;
import java.util.List;
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

public class task1_1_1 {

	public static class KeyMapper
			extends Mapper<Object, Text, Text, IntWritable> {

		private static final IntWritable one = new IntWritable(1);
		private Text word = new Text();
		private Set<String> stopWordList = new HashSet<String>();
		private BufferedReader fis;

		
		@SuppressWarnings("deprecation")
		protected void setup(Context context) throws java.io.IOException,
				InterruptedException {
			try {

				Path[] stopWordFiles = new Path[0];
				stopWordFiles = context.getLocalCacheFiles();
				System.out.println(stopWordFiles.toString());
				if (stopWordFiles != null && stopWordFiles.length > 0) {
					for (Path stopWordFile : stopWordFiles) {
						readStopWordFile(stopWordFile);
					}
				}
			} catch (IOException e) {
				System.err.println("Exception reading stop word file: " + e);

			}
		}

		private void readStopWordFile(Path stopWordFile) {
			try {
				fis = new BufferedReader(new FileReader(stopWordFile.toString()));
				String stopWord = null;
				while ((stopWord = fis.readLine()) != null) {
					stopWordList.add(stopWord);
				}
			} catch (IOException ioe) {
				System.err.println("Exception while reading stop word file '"
						+ stopWordFile + "' : " + ioe.toString());
			}
		}

		public static String getFileName(String filePath) {
			// Split the path by "/"
			String[] pathParts = filePath.split("/");

			// The filename will be the last element
			String fileName = pathParts[pathParts.length - 1];
			String folderName = pathParts[pathParts.length - 2];

			// Remove the extension (".txt") using substring
			int extensionIndex = fileName.lastIndexOf(".");
			if (extensionIndex > 0) {
				fileName = fileName.substring(0, extensionIndex);
			}

			return folderName + "." + fileName;
			// return fileName;
		}

		public static String removeSpecialChars(String text) {
			return text.replaceAll("[^\\w\\s]", "");
		}

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

			setup(context);
			InputSplit split = context.getInputSplit();
			Path path = ((FileSplit) split).getPath();

			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				String token = removeSpecialChars(itr.nextToken().toLowerCase());

				if (stopWordList.contains(token) | token.isEmpty()) {
					continue;
				} else {
					word.set(token);
					context.write(word, one);
				}
			}
		}
	}

	public static class KeyReducer
			extends Reducer<Text, IntWritable, Text, IntWritable> {
		private Integer termcnt;

		@Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
			termcnt = 1;
        }
		
		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			context.write(key, new IntWritable(termcnt++));
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "hash termid");
		job.setJarByClass(task1_1_1.class);
		job.setMapperClass(KeyMapper.class);
		job.setReducerClass(KeyReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.getConfiguration().set("mapreduce.output.basename", "termid.mtx");
		FileInputFormat.setInputDirRecursive(job, true);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		job.addCacheFile(new Path(args[1]).toUri());
		FileOutputFormat.setOutputPath(job, new Path(args[2]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
