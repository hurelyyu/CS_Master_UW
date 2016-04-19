

import java.io.*;

public class testMain {
	
	public static void main(String[] args) {
		
		String myProtocal = null;
		if (args[0].equals("TCP")){
			myProtocal = "TCP";
		} else if (args[0].equals("UDP")) {
			myProtocal = "UDP";
		} else {
			System.out.println("Usage: testMain <protocol: either TCP or UDP>");
			System.exit(0);
		}

		try {
			File fl = new File("testPairs.txt");
			InputStreamReader read = new InputStreamReader(new FileInputStream(fl));
			BufferedReader br = new BufferedReader(read);
			String line;
			
			while ((line = br.readLine()) != null) {
				System.out.println("Input line: " + line);
				oneExecution(line, myProtocal);
			}
			read.close();
			System.out.println("Test finished. ");
		} catch (IOException e) {
		}
	
	}
	public static void oneExecution(String clientInputArgs, String myProtocal) {
		String[] clientArgs = clientInputArgs.split(" ");

		if (myProtocal.equals("TCP")){
			TCPClientMain.main(clientArgs);
		} else if (myProtocal.equals("UDP")) {
			UDPClientMain.main(clientArgs);	
		} else {
			System.out.println("Usage: testMain <TCP or UDP>");
		}
	}
	
	/*
	public static void main(String[] args) {
		try {
			ProcessBuilder pb = new ProcessBuilder("/bin", "-c", "/testPairs.sh");
			final Process process = pb.start();
			BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line;
			int count = 0;
			
			while ((line = br.readLine()) != null) {
				System.out.println("line: " + line);
				count  = count + 1 ;
			}
			System.out.println("Test finished. " + count + "lines executed.");
		} catch (IOException e) {
			
		}	
	
	}*/
}
