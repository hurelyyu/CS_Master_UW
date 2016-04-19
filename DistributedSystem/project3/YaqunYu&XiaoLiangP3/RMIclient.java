import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.net.Inet4Address;


public class RMIclient implements Runnable {
	public static int timeout = 1000*3;
	public static int id;
	static RMIinterface rmiitf ;
	
	public static void main(String args[]) throws Exception{
		//RMIinterface rmiitf = (RMIinterface) Naming.lookup("RMIinterface");
		try {System.out.println("!!!!!!!!!!!!!!###########");
			myLog.createLog("RMIClientLog.txt");
			}catch (IOException e) {
			System.out.println("Error: Log creation for RMI server failed.");
			myLog.error("Error: Log creation for RMI server failed.");
			}
		  String myHost = "";
	        if(args.length != 1) {System.out.println("!!!!!!!!!!!!!!");
	        	myLog.error("Server IP is not provided as a parameter, if your want to use default, please input default ip address: 127.0.0.1");
	        	//myHost = "127.0.0.1";
	        } else {System.out.println("###########");
	        	myHost = java.net.Inet4Address.getLocalHost().getHostName();
	        	myLog.normal("Will call server: " + args[0], true);
	        }
		     
		try{
			System.out.println("!!!!!!!!!!!!!!###########");
	        	//Registry registry = LocateRegistry.getRegistry(myHost); //get registry
	        rmiitf = (RMIinterface)Naming.lookup("//" + args[0] + "/"+ "RMIinterface"); //obtain the stub for the remote object from the server's registry
	        	//System.out.println("connect Success");  //Debug Measurement 2
	    	}catch (Exception e){
			myLog.error("Client connect with Server Error: " + e.getMessage());
			System.exit(-1);
	    	}
		//setup timestamp
		String timestamp = myLog.getTimestamp();
		myLog.normal("RMIClient-" + id +"-" + timestamp + " Log.txt", true);
		myLog.normal("Client " + id + " is running on : " 
				+ Inet4Address.getLocalHost(), true);
		//open file and read line by line
		try {
            File fl = new File("testPairs.txt");
            InputStreamReader read = new InputStreamReader(new FileInputStream(fl));
            BufferedReader br = new BufferedReader(read);
            String line;
            
            while ((line = br.readLine()) != null) {
                System.out.println("Input line: " + line);
                oneExecution(line, args[0]);
            }
            read.close();
            System.out.println("Test finished. ");
        } catch (IOException e) {
        	myLog.error("Could not find file name testPairs.txt, please make sure file location is correct");
        }
	}	

	public static void oneExecution(String clientInputArgs, String serverIP) throws RemoteException, MalformedURLException, NotBoundException {
		 System.out.println("##@@@@@@@@@@@@@@@@##");  
		String[] clientArgs = clientInputArgs.split("/");
			 
			  String operation = clientArgs[0];	
			  System.out.println("#######" + operation + "####");			
			  String thekey  ="";
			  //System.out.print ("Value is : ");
			  String thevalue  ="";
		//put
		if (operation.equals("put")){	
		thevalue = clientArgs[2];
		thekey  = clientArgs[1];
	    try{rmiitf.put(thekey, thevalue) ;
				myLog.normal("Request succesfully executed : PUT (" + thekey + ", " + thekey + ")", true);
		}catch(Exception e){
				myLog.error("unknown error: " + e.getMessage());
			 System.out.println();
			 try {
				Thread.sleep(timeout);
			} catch (InterruptedException e1) {
				// TODO Auto-generated catch block
				myLog.error("Error: " + e1.getMessage());
			  }
		   }
		}	
		//get function
		else if(operation.equals("get")){	 
			 
			  thekey  = clientArgs[1];
			  try{rmiitf.get(thekey);
			  if (rmiitf.get(thekey) != null) {
					myLog.normal("Request succesfully executed: " + "GET thekey: " + thekey + " and its Value: " + rmiitf.get(thekey), true);
				} else {
					myLog.error("GET request with invalid thekey: " + "thekey " + thekey + " is not in set.");
				}
			  }catch(Exception e){
				  myLog.error("unknown error: " + e.getMessage());
			 try{
				Thread.sleep(timeout);
			} catch (InterruptedException e1) {
				// TODO Auto-generated catch block
				myLog.error("Error: " + e1.getMessage());
			}
			}
		}	
		//delete function
		else if(operation.equals("delete"))  {
			
			thekey  = clientArgs[1]; 	
			 try{rmiitf.delete(thekey);
			  myLog.normal("RMIClient " + id 
						+ " has issued request: delete "+ thekey  , true);
			 }catch(Exception e){
				 myLog.error("unknown error: " + e.getMessage());
			 try {
				Thread.sleep(timeout);
			} catch (InterruptedException e1) {
				// TODO Auto-generated catch block
				myLog.error("Error: " + e1.getMessage());
			}
		}
	}
	else{
  	myLog.normal("All finished",true);
  }
  }
  @Override
	public void run() {
		// TODO Auto-generated method stub
		
	}
}
