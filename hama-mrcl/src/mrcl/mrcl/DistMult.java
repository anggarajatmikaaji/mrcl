package mrcl;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileAsBinaryOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;

public class DistMult {
	public static void main(String[] args) {
		new DistMult().run();
	}
	
	public void run(){
		try {
			Configuration conf = new Configuration(true);
			int n = 100;
			Matrix a = Matrix.createRandomRemote("a", n, n, 1, conf);
			Matrix b = Matrix.createRandomRemote("b", n, n, 2, conf);
			String jobName = makeJob(a, b, conf);
			
			JobConf job = new JobConf(DistMult.class);
			job.setMapperClass(MultMap.class);
			job.setReducerClass(MultReduce.class);
			job.setCombinerClass(MultReduce.class);
			job.setInputFormat(TextInputFormat.class);
			FileInputFormat.setInputPaths(job, new Path(jobName));
			FileSystem fs = FileSystem.get(conf);
			Path outDir = new Path("some");
			if (fs.exists(outDir))
				fs.delete(outDir, true);
			FileOutputFormat.setOutputPath(job, outDir);
			
			JobClient.runJob(job);
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	public String makeJob(Matrix a, Matrix b, Configuration conf) {
		try {
			FileSystem fs = FileSystem.get(conf);
			String jobName = String.format(
					"mrcl/jobs/%s__mult__%s", a.getName(), b.getName());
			DataOutputStream dos = fs.create(new Path(jobName));

			int rounds = a.getBlockCols();
			for (int round = 0; round < rounds; round++)
				dos.writeUTF(new MultArgs(a.getName(), b.getName(), round)
						+ "\n");
			
			return jobName;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static class MultArgs implements Writable {
		String _a;
		String _b;
		int _round;

		public MultArgs(String string) {
			String[] words = string.split("/");
			_a = words[0];
			_b = words[1];
			_round = Integer.parseInt(words[2]);
		}

		public MultArgs(String a, String b, int round) {
			_a = a;
			_b = b;
			_round = round;
		}

		public String toString() {
			return String.format("%s__%s__%d", _a, _b, _round);
		}

		public String getA() {
			return _a;
		}

		public String getB() {
			return _b;
		}

		public int getRound() {
			return _round;
		}

		@Override
		public void readFields(DataInput input) throws IOException {
			_a = input.readUTF();
			_b = input.readUTF();
			_round = input.readInt();
		}

		@Override
		public void write(DataOutput output) throws IOException {
			output.writeUTF(_a);
			output.writeUTF(_b);
			output.writeInt(_round);
		}
	}

	public static class MultMap implements
			Mapper<LongWritable, Text, MultArgs, Matrix> {

		private JobConf conf;

		@Override
		public void map(LongWritable lineNo, Text line,
				OutputCollector<MultArgs, Matrix> output, Reporter reporter)
				throws IOException {

			MultArgs args = new MultArgs(line.toString());
			FileSystem f = FileSystem.get(conf);

			DataInputStream aIs = f.open(new Path(Matrix.getPath(args.getA())));
			DataInputStream bIs = f.open(new Path(Matrix.getPath(args.getB())));

			Matrix a = Matrix.read(aIs);
			Matrix b = Matrix.read(bIs);

			aIs.close();
			bIs.close();

			Matrix inter = Matrix.multiplyRemote("c", a, b, args.getRound(),
					conf);

			output.collect(new MultArgs(a.getName(), b.getName(), 0), inter);
		}

		@Override
		public void configure(JobConf conf) {
			this.conf = conf;
		}

		@Override
		public void close() throws IOException {
		}

	}

	public static class MultReduce implements
			Reducer<MultArgs, Matrix, MultArgs, Matrix> {
		private Configuration conf;

		@Override
		public void reduce(MultArgs key, Iterator<Matrix> values,
				OutputCollector<MultArgs, Matrix> output, Reporter reporter)
				throws IOException {
			Matrix value = values.next();
			Matrix sum = Matrix.createFillRemote(key.toString(), value
					.getRows(), value.getCols(), 0, conf);
			Matrix.addRemote(sum.getName(), sum, value, conf);

			while (values.hasNext()) {
				value = values.next();
				Matrix.addRemote(sum.getName(), sum, value, conf);
			}

			output.collect(key, sum);
		}

		@Override
		public void configure(JobConf conf) {
			this.conf = conf;
		}

		@Override
		public void close() throws IOException {
		}
	}
}
