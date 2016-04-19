import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Date;
import java.text.SimpleDateFormat;

public class serverlog {
	protected static int portNumber;
	protected static String requesttime;
	protected static String request;
	String reply;
	serverlog(String reply){
		//create client log file as .txt
		String fileName = "serverlog.txt";
		File fl = new File(fileName);
		Long currenttime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
		//Date resulttime = new Date(currenttime);
		String date = sdf.format(new Date(currenttime));
		if(fl.exists() && fl.isDirectory()){
			try{
				FileWriter fw = new FileWriter (fl);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.write("\n"+ "Portnumber is: " +"\t" + portNumber);
				bw.write("\n" + "Request time is:  " + request + "\t" + requesttime);
    			bw.write("\n" + "Reply" + reply + "\t" + date);
    			System.out.println(request + "\t" + requesttime );
				System.out.println(reply + "\t" + date);
				bw.close();
				System.out.println("Server log File written Succesfully");
			}catch(IOException e){
				//print to the standard error
				System.err.println("Server log File Fail to written because: " + e.getMessage());
			}
		}else{ 
			try {
				PrintWriter pw = new PrintWriter(fileName);
				pw.println("Portnumber is: " +"\t" + portNumber);
				pw.write("\n" + "Request time is:  " + request + "\t" + requesttime);
    			pw.write("\n" + "Reply" + reply + "\t" + date);
				System.out.println(reply + "\t" + date);
				pw.close();
				System.out.println("Server log file generated as: " + fileName);
				
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				
				System.err.println("Server log File Fail to generate because: " + e.getMessage());
			}
			
		}
	}
}
