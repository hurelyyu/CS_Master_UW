import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Date;
import java.text.SimpleDateFormat;

public class clientlog{
		String output1;
		int output2;
		clientlog(String output1, int output2) {
			this.output1 = output1;
			this.output2 = output2;
		//create client log file as .txt
		String fileName = "clientlog.txt";
		File fl = new File(fileName);
		Long currenttime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
		//Date resulttime = new Date(currenttime);
		String date = sdf.format(new Date(currenttime));
		//if time out, send out time out err message
		System.err.println("Timed out to connect server at port: "+output2+" "+date);
		if(fl.exists() && fl.isDirectory()){
			try{
				FileWriter fw = new FileWriter (fl);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.write(output1 +"\t" + output2 +"\t" + date);
				System.out.println(output1 +"\t" + output2 +"\t" + date);
				bw.close();
				System.out.println("Client log File written Succesfully");
			}catch(IOException e){
				//print to the standard error
				System.err.println("Client log File Fail to written because: " + e.getMessage());
			}
		}else{ 
			try {
				PrintWriter pw = new PrintWriter(fileName);
				pw.println(output1 +"\t" + output2 +"\t" + date);
				System.out.println(output1 +"\t" + output2 +"\t" + date);
				pw.close();
				System.out.println("Client log file generated as: " + fileName);
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				
				System.err.println("Client log File Fail to generate because: " + e.getMessage());
			}
			
		}
	}
}

