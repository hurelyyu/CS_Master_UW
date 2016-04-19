import java.io.IOException;
import java.net.Inet4Address;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
private static int NUM_CLIENTS = 2;
	private int sleeptime = 1000;
	private RMIinterface rmiitf;
	private int myid;
	private String myHost;

public static void main(String[] args) throws NotBoundException, IOException {
		if(args.length<1)
			myLog.error("Usage: Java RMIClient <host>");
			//System.exit(1);
		for(int ,myid=1; myid <= NUM_CLIENTS; myid ++){
			(new Thread(new RMIclient2(args[0], myid))).start();
		}
		
	}

public class RMIclient2 implements Runnable {


	public RMIclient2(String Host, int id) throws NotBoundException, IOException{
		String name = "RMIinterface";
		myid = id;
		myHost = java.net.Inet4Address.getLocalHost().getHostName();
		try{
			rmiitf = (RMIinterface)Naming.lookup("//" + Host + "/" + name);
		}catch(Exception e){
			myLog.error("Error: " + e.getMessage());
		}
		String timestamp = myLog.getTimestamp();
		myLog.normal("Client-" + id +"-" + timestamp + ".log", true);
		myLog.normal("Client " + id + " is running on : " 
				+ Inet4Address.getLocalHost(), true);
	}



	private void put(String key, String value) throws RemoteException, InterruptedException{
		rmiitf.put(key, value);
		myLog.normal(myHost + ": Thread " 
				+ myid + ": put " + key + " " + value, true);
		myLog.normal("Client " + myid 
				+ " has issued request: put "+ key+","+ value, true);
		Thread.sleep(sleeptime);
	}
	
	private void get(final String key) throws RemoteException, InterruptedException{
		String v;
		myLog.normal("Client " + myid 
				+ " has issued request: get "+ key, true);
		v = rmiitf.get(key);
		myLog.normal(myHost + ": Thread " 
				+ myid + ": get " + key + " " + v, true);
		myLog.normal("Client " + myid + " has got--> key = "
				+ key+ " , value = " + v, true);
		Thread.sleep(sleeptime);
	}
	private void delete(final String key) throws InterruptedException, RemoteException {
		rmiitf.delete(key);
		myLog.normal(myHost + ": Thread " 
				+ myid + ": delete " + key, true);
		myLog.normal("Client " + myid 
				+ " has issued request: delete "+ key, true);
		Thread.sleep(1000);		
	}
	public void run() {
		// TODO Auto-generated method stub
		while(true) {
			try{
				put("key1", "value1");
				put("key2", "value2");
				put("key3", "value3");
				put("key4", "value4");
				put("key5", "value5");

				get("key1");
				get("key2");
				get("key3");
				get("key4");
				get("key5");

				delete("key1");
				delete("key2");
				delete("key3");
				delete("key4");
				delete("key5");

				get("key1");

			}catch(Exception e){
				myLog.error("Error in running: " + e.getMessage());
			}
			
		}
	}	
}
