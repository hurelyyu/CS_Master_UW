// updateclientlog.java is the java file to generate/update the client's log 
import java.io.*;
import java.util.Date;
import java.text.SimpleDateFormat;

public class UpdateClientLog {
	String message;
	UpdateClientLog(String message) {
		this.message = message;
		File f = new File("clientlog.txt");
		long curtime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
		Date resultdate = new Date(curtime);
		String datetime = sdf.format(resultdate);
		if(f.exists() && !f.isDirectory()) { 
			try {
				FileWriter fstream = new FileWriter("clientlog.txt", true); //true tells to append data.
    			BufferedWriter out = new BufferedWriter(fstream);
    			out.write("\n" + message + "\t" + datetime);
    			out.close();
    			System.out.println("Client log is updated as clientlog.txt");
			}
			catch (IOException ioe)
			{
    			System.err.println("Error: The log is not able to be generated");
			}
		}
		else {
			try {
				PrintWriter writelog = new PrintWriter("clientlog.txt", "UTF-8");
				writelog.println(message + "\t" + datetime);
				writelog.close();
				System.out.println("Client log is generated as clientlog.txt");
			}
			catch (IOException ioe){
				System.err.println("Error: the log is not able to be generated");
			}
		}
	}

}