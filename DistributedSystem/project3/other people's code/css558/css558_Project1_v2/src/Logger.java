import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;


public class Logger {

	private PrintWriter my_writer;
	
	public Logger(final String the_file_name) throws IOException {
		File f = new File(the_file_name);
		if(!f.exists()) {
			f.createNewFile();
		}
		my_writer = new PrintWriter(new BufferedWriter(new FileWriter(f)));
	}
	
	public void write(final String the_line) {
		my_writer.println(System.currentTimeMillis() + ": " + the_line);
	}
	
	public void close() {
		my_writer.close();
	}
}
