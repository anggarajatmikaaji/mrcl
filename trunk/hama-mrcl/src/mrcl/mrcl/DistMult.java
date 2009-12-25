package mrcl;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.FloatBuffer;
import java.util.Iterator;

import mrcl.lib.Block;
import mrcl.lib.Matrix;
import mrcl.lib.MultArgs;

import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

@SuppressWarnings("deprecation")
public class DistMult extends Configured implements Tool {
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new DistMult(), args);
	}

	@Override
	public int run(String[] args) throws Exception {
		try {
		    Configuration conf = getConf();
			JobConf job = new JobConf(conf, DistMult.class);
			
			boolean useJCublas = job.getBoolean("useJCublas", false); // Example: -DuseJCublas=true
			int n = job.getInt("matrix.size", 1000); // Example: -Dmatrix.size=100000
			Block.BLOCK_SIZE = job.getInt("block.size", 1024); // Example: -Dblock.size=2048
			
			Matrix a = Matrix.createRandomRemote("bb", n, n, 1, conf);
			a.writeRemote(conf);
			Matrix b = Matrix.createRandomRemote("cc", n, n, 2, conf);
			b.writeRemote(conf);
			String jobName = makeJob(a, b, conf);

			job.setJobName("MM-" + (useJCublas ? "JCublas" : "Java"));
			job.setMapperClass(MultMap.class);
			job.setReducerClass(MultReduce.class);
			job.setCombinerClass(MultCombine.class);
			job.setInputFormat(TextInputFormat.class);
			job.setMapOutputKeyClass(MultArgs.class);
			job.setMapOutputValueClass(Matrix.class);
			job.setOutputKeyClass(MultArgs.class);
			job.setOutputValueClass(Matrix.class);
			FileInputFormat.setInputPaths(job, new Path(jobName));
			FileSystem fs = FileSystem.get(conf);
			Path outDir = new Path("some");
			if (fs.exists(outDir))
				fs.delete(outDir, true);
			fs.close();
			FileOutputFormat.setOutputPath(job, outDir);

			JobClient.runJob(job).waitForCompletion();

			if (job.getBoolean("validate", false)) { // Example: -Dvalidate=true
				FloatBuffer distResult = Matrix.readRemote("result", conf)
						.getFloatBufferRemote(conf);
	
				Matrix c = Matrix.createRandomLocal("c", n, n, 1);
				Matrix d = Matrix.createRandomLocal("d", n, n, 2);
				Matrix e = Matrix.multiplyLocal("e", c, d);
				FloatBuffer localResult = e.getFloatBufferLocal();
	
				for (int i = 0; i < 100; i++) {
					System.out.printf("%f, %f\n", distResult.get(i), localResult
							.get(i));
				}
			}

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		return 0;
	}

	public String makeJob(Matrix a, Matrix b, Configuration conf) {
		try {
			FileSystem fs = FileSystem.get(conf);
			String jobName = String.format("/mrcl/jobs/mult/%s/%s",
					a.getName(), b.getName());
			FSDataOutputStream dos = fs.create(new Path(jobName));
			int rounds = a.getBlockCols();
			StringBuilder builder = new StringBuilder();
			for (int round = 0; round < rounds; round++)
				builder.append(
						new MultArgs(a.getName(), b.getName(), round)
								.toString()).append('\n');
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(dos
					.getWrappedStream()));
			bw.write(builder.toString());
			bw.close();
			dos.close();
			fs.close();
			return jobName;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static class MultMap implements
			Mapper<LongWritable, Text, MultArgs, Matrix> {

		private Configuration conf;

		@Override
		public void map(LongWritable lineNo, Text line,
				OutputCollector<MultArgs, Matrix> output, Reporter reporter)
				throws IOException {

			MultArgs args = new MultArgs(line.toString());
			reporter.setStatus("read matrices");
			Matrix a = Matrix.readRemote(args.getA(), conf);
			Matrix b = Matrix.readRemote(args.getB(), conf);
			
			reporter.setStatus("multiply");
			Matrix inter = Matrix.multiplyRemote(a.getName() + "_"
					+ b.getName(), a, b, args.getRound(), conf);
			reporter.setStatus("write");
			inter.writeRemote(conf);

			reporter.setStatus("collect");
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

	public static class MultCombine implements
			Reducer<MultArgs, Matrix, MultArgs, Matrix> {
		private Configuration conf;

		@Override
		public void reduce(MultArgs key, Iterator<Matrix> values,
				OutputCollector<MultArgs, Matrix> output, Reporter reporter)
				throws IOException {
			Matrix value = values.next();
			Matrix sum = Matrix.createFillRemote("/__tmp/sum/"
					+ value.getName(), value.getRows(), value.getCols(), 0,
					conf);

			sum.writeRemote(conf);
			sum = Matrix.addRemote("/__tmp/sum/" + value.getName(), sum, value,
					conf);
			sum.writeRemote(conf);
			reporter.progress();

			while (values.hasNext()) {
				reporter.progress();
				value = values.next();
				sum = Matrix.addRemote("/__tmp/sum/" + value.getName(), sum,
						value, conf);
				sum.writeRemote(conf);
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

	public static class MultReduce implements
			Reducer<MultArgs, Matrix, MultArgs, Matrix> {
		private Configuration conf;

		@Override
		public void reduce(MultArgs key, Iterator<Matrix> values,
				OutputCollector<MultArgs, Matrix> output, Reporter reporter)
				throws IOException {
			Matrix a = Matrix.readRemote(key.getA(), conf);
			Matrix sum = Matrix.createFillRemote("/__tmp/base", a.getRows(), a
					.getCols(), 0, conf);
			while (values.hasNext()) {
				Matrix value = values.next();
				if (!values.hasNext())
					sum = Matrix.addRemote("result", sum, value, conf);
				else
					sum = Matrix.addRemote("/__tmp/sum/" + value.getName(),
							sum, value, conf);
				sum.writeRemote(conf);
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
