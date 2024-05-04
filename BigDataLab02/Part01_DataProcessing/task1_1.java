
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.TreeMap;
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

public class task1_1 {

	public static class KeyMapper
			extends Mapper<Object, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();
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
				Path stopWordFile = cachedFiles[0];
				Path termidFile = cachedFiles[1];
				Path docidFile = cachedFiles[2];

				if (stopWordFile != null) {
					readStopWordFile(stopWordFile);
				}

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


		void readTermIDFile(Path termidPath){
			try {
				fis = new BufferedReader(new FileReader(termidPath.toString()));
				String terminfo = null;
				while ((terminfo = fis.readLine()) != null) {
					String[] term_and_id = terminfo.split("\t");
					termid.put(term_and_id[0], term_and_id[1]);
				}
			} catch (IOException ioe) {
				System.err.println("Exception while reading termid file '"
						+ termidPath + "' : " + ioe.toString());
			}
		}

		void readDocIDFile(Path docidPath){
			try {
				fis = new BufferedReader(new FileReader(docidPath.toString()));
				String docinfo = null;
				while ((docinfo = fis.readLine()) != null) {
					String[] doc_and_id = docinfo.split("\t");
					docid.put(doc_and_id[0], doc_and_id[1]);
				}
			} catch (IOException ioe) {
				System.err.println("Exception while reading docid file '"
						+ docidPath + "' : " + ioe.toString());
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

				if (stopWordList.contains(token)) {
					continue;
				} else {

					if(docid.containsKey(getFileName(path.toString())) && termid.containsKey(token)){
						word.set(docid.get(getFileName(path.toString())) + "\t" + termid.get(token));
					}
					context.write(word, one);
				}
			}
		}
	}

	public static class KeyReducer
			extends Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "hased word count");
		job.setJarByClass(task1_1.class);
		job.setMapperClass(KeyMapper.class);
		job.setReducerClass(KeyReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.getConfiguration().set("mapreduce.output.basename", "task_1_1.mtx");
		FileInputFormat.setInputDirRecursive(job, true);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		job.addCacheFile(new Path(args[1]).toUri());
		job.addCacheFile(new Path(args[2]).toUri());
		job.addCacheFile(new Path(args[3]).toUri());
		FileOutputFormat.setOutputPath(job, new Path(args[4]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
