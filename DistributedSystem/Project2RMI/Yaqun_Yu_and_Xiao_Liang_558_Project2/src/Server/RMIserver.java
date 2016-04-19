import java.io.IOException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashMap;

public class RMIserver /*extends UnicastRemoteObject */implements RMIinterface
{


	protected RMIserver() throws RemoteException {
		//super();
		// TODO Auto-generated constructor stub
	}
	public HashMap<Object, Object> pairs = new HashMap<Object, Object>();
	
	//Define put operator
	public int put(Object key, Object value) throws RemoteException {
		// TODO Auto-generated method stub
		pairs.put(key,value);
		myLog.normal("Key: " + key +", "+ "Value: " + value + " pair is added");
		//System.out.println(key+", "+value+" pair is added");
		//myLog.normal("Request from " + ip + ": " + port +" succesfully executed : PUT (" + key + ", " + value + ")");
		//return 0;
		return 0;
	}
	
	//Define get operator
	public Object get(Object key) throws RemoteException {
		// TODO Auto-generated method stub
		if(!pairs.containsKey(key))
		{
			myLog.error("Get Key: " + key + " does not exist in KeySet");
			return null;
		}else{
			myLog.normal("key get from system is: " + key + ", and its value is: " + pairs.get(key));
			return pairs.get(key);
		}
	}
	
	//Define delete operator
	public int delete(Object key) throws RemoteException {
		// TODO Auto-generated method stub
		if(!pairs.containsKey(key))
			{
			myLog.error("Delete Key: " + key + " does not exist in KeySet");
			return 1;
			}
			//try {
				//throw new MyException("Error: "+key+" does not exist in KeySet");
			//} catch (MyException e) {
				// TODO Auto-generated catch block
				//e.printStackTrace();
			//}
		
		else {
			pairs.remove(key);
			myLog.normal(key + " has been removed from set along with its value");
			return 0;
		}
		
		
	}
	
	//Main function
	public static void main(String args[]) throws Exception{
		//Open log
		try {
			myLog.createLog("RMIServerLog.txt");
		}  catch (IOException e) {
			System.out.println("Error: Log creation for RMI server failed.");
		}
	    int serverPort = 0;
	    try {
			serverPort = Integer.parseInt(args[0]);
			} catch (Exception e) {
				myLog.error("RMIServer started failed: arguments error.");
				System.out.println("Usage: RMIServerMain <port number>");
				System.exit(0);
			}
	    
		  try{

			RMIserver tcpsv = new RMIserver();
			RMIinterface stub = (RMIinterface) UnicastRemoteObject.exportObject(tcpsv,0);
			Registry registry = LocateRegistry.getRegistry(serverPort);
			registry.rebind("RMIinterface", stub);
			//System.out.println("Server bound....");
			myLog.normal("Server bound......");
		  }catch(RemoteException e){
		         myLog.error("Remote Exception");
		         System.out.println("Remote: " + e.getMessage());
		  }catch (Exception e) {
					myLog.error("Exception");
		         System.out.println("Exception: " + e.getMessage());
					System.exit(1);  
		  }
	}
}
