package util;

/**
 * Created by warn on 21/4/2016.
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class HdfsWriter extends Configured implements Tool {

    private static final String FS_PARAM_NAME = "fs.defaultFS";

    public static void main(String[] args) throws Exception {
        int returnCode = ToolRunner.run(new HdfsWriter(), args);
        System.exit(returnCode);
    }

    public int run(String[] args) throws Exception {

        if (args.length < 2) {
            System.err.println("HdfsWriter [local input path] [hdfs output path]");
            return 1;
        }

        String localInputPath = args[0];
        Path outputPath = new Path(args[1]);

        Configuration conf = getConf();
        System.out.println("configured filesystem = " + conf.get(FS_PARAM_NAME));
        System.out.println("Save file to " + outputPath.toString());
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }
        OutputStream os = fs.create(outputPath);
        InputStream is = new BufferedInputStream(new FileInputStream(localInputPath));
        IOUtils.copyBytes(is, os, conf);
        System.out.println("File " + localInputPath + " update succeeded to " + outputPath);
        return 0;
    }
}
