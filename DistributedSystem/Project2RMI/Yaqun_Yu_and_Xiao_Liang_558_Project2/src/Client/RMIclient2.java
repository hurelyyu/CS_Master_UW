import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.logging.Logger;

public class RMIclient2 {
	
	public static void main(String args[]) throws Exception{
		//RMIinterface rmiitf = (RMIinterface) Naming.lookup("RMIinterface");
		try {
			myLog.createLog("RMIClientLog.txt");
		}  catch (IOException e) {
			System.out.println("Error: Log creation for RMI server failed.");
			myLog.error("Error: Log creation for RMI server failed.");
		}
		  String myHost = "";
	        if(args.length != 1) {
	        	myLog.error("Server IP is not provided as a parameter, if your want to use default, please input default ip address: 127.0.0.1");
	        	//myHost = "127.0.0.1";
	        } else {
	        	myHost = args[0];
	        	myLog.normal("Will call server: " + myHost);
	        }
		     /*   String myProtocal = null;
		        if (args[0].equals("RMI")){
		            myProtocal = "RMI";
		        } else {
		            System.out.println("Usage: testMain <protocol: Please Enter RMI only>");
		            System.exit(0);
		        } */
		 //Check if communication is correct
		RMIinterface rmiitf = null;
		try{
	        	Registry registry = LocateRegistry.getRegistry(myHost); //get registry
	        	rmiitf = (RMIinterface)registry.lookup("RMIinterface"); //obtain the stub for the remote object from the server's registry
	        	//System.out.println("connect Success");  //Debug Measurement 2
	    	}catch (Exception e){
			myLog.error("Client connect with Server Error: " + e.getMessage());
			System.exit(-1);
		}
		//open file and read line by line
		try {
            File fl = new File("testPairs.txt");
            InputStreamReader read = new InputStreamReader(new FileInputStream(fl));
            BufferedReader br = new BufferedReader(read);
            String line;
            
            while ((line = br.readLine()) != null) {
                System.out.println("Input line: " + line);
                oneExecution(line, myHost);
            }
            read.close();
            System.out.println("Test finished. ");
        } catch (IOException e) {
        	myLog.error("Could not find file name testPairs.txt, please make sure file location is correct");
        }
	}	

	public static void oneExecution(String clientInputArgs, String serverIP) throws RemoteException, MalformedURLException, NotBoundException {
		// TODO Auto-generated method stub
		//System.out.println("it is here 1 !!!"); Debug Measurement 1
		RMIinterface rmiitf = null;
		try{
	        	Registry registry = LocateRegistry.getRegistry(serverIP); //get registry
	        	rmiitf = (RMIinterface)registry.lookup("RMIinterface"); //obtain the stub for the remote object from the server's registry
	        	//System.out.println("connect Success");  //Debug Measurement 2
	    	}catch (Exception e){
			myLog.error("Client connect with Server Error: " + e.getMessage());
			System.exit(-1);
		}
		//RMIinterface rmiitf = (RMIinterface) Naming.lookup("RMIinterface");
		String[] clientArgs = clientInputArgs.split("/");
		switch (clientArgs[0]){
		case "put":
			 
			  //System.out.print ("Key is : ");
			  Object key  = clientArgs[1];
			  //System.out.print ("Value is : ");
			  Object value  = clientArgs[2];
			  // Call remote method
			  //System.out.println 
			  //("key and value is put into system : " + rmiitf.put(key, value));
			  //System.out.println("****GET_K****:" + key + "*");
			  //System.out.println("****PUT_V****" + value + "*");
			 try{ int resultNum1 = rmiitf.put(key, value);
			 // myLog.normal("Request from " +  ": " + args[0] +" succesfully executed : PUT (" + key + ", " + value + ")");
			  if (resultNum1 == 0) {
					//result = "PUT: ("+key+","+value+") Successfully operated.";
					myLog.normal("Request succesfully executed : PUT (" + key + ", " + value + ")");
				} else {
					//result = "Error: PUT operation failed. ";
					myLog.error("unknown error or received PUT request with invalid KEY and VALUE");
		    
				}
			}catch(Exception e){
				myLog.error("unknown error: " + e.getMessage());
			}
			  System.out.println();
			 break;
		   
			case "get":
			  
			  System.out.print ("key : ");
			  key = clientArgs[1];
			  // Call remote method
			 /*System.out.println 
		     ("key get from system is: " + rmiitf.get(key));
			 System.out.println("****GET_K****:" + key + "*"); */
			 Object resultNum2 = rmiitf.get(key);
			 // myLog.normal("Request from " +  ": " + args[0] +" succesfully executed : PUT (" + key + ", " + value + ")");
			  if (resultNum2 != null) {
					//result = "PUT: ("+key+","+value+") Successfully operated.";
					myLog.normal("Request succesfully executed: " + "GET Key: " + key + " and its Value: " + resultNum2);
				} else {
					//result = "Error: PUT operation failed. ";
					myLog.error("GET request with invalid KEY: " + "Key " + key + " is not in set.");
		    
				}
			break;
			case "delete":
			
			  System.out.print ("key : ");
			  key = clientArgs[1];
			  // Call remote method
			 // System.out.println 
		      //("key delete from system is: " + rmiitf.delete(key));
			  //rmiitf.delete(key1);
			 // myLog.normal("Request successfully executed : DELETE VALUE of KEY " + key1);
			  int resultNum3 = rmiitf.delete(key);
				  if (resultNum3 == 0) {
						//result = "PUT: ("+key+","+value+") Successfully operated.";
						myLog.normal("Request succesfully executed: " + "DELETE Key: " + key);
					} else {
						//result = "Error: PUT operation failed. ";
						myLog.error("DELETE request with invalid KEY: " + "Key " + key + " is not in set.");
			    
					}
			break;
			case "exit":
			  System.exit(0);
			default :
			  System.out.println ("Invalid option");
			break;
			}
			
		}
}

