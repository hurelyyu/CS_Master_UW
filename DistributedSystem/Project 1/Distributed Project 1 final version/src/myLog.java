

import java.text.*;
import java.util.Date;
import java.io.FileWriter;
import java.io.IOException;

public class myLog {
	
	private static FileWriter fw;
	private static String logTimer() {
		SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy hh:mm:ss:SSS");
		Date currentTime = new Date(System.currentTimeMillis());
		String currentTimeString = formatter.format(currentTime);
		return currentTimeString;
	}
	
	public static void createLog(String path) throws IOException {
		fw = new FileWriter(path, true);
	}

	public static void normal(String logContent) {
		System.out.println(logTimer() + logContent);
		try {
			fw.write(logTimer() + "  " + "[Message]" +  logContent + "\r\n");
			fw.flush();
		} catch (IOException e) {
		}
	}

	public static void error(String logContent) {
		System.out.println(logTimer() + logContent);
		try {
			fw.write(logTimer() + "  " + "[Error]" + logContent + "\r\n");
			fw.flush();
		} catch (IOException e) {
		}
	}
}
