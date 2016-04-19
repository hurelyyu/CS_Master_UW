
	import java.rmi.Remote;
	import java.rmi.RemoteException;
	import java.util.HashMap;

	public interface RMIinterface extends Remote{

	  // public HashMap<Object, Object> pairs = new HashMap<Object, Object>();
		// put, get, delete
		public void put(String thekey, String thevalue) throws RemoteException;//{
			// TODO Auto-generated method stub
			//return ;
		//}
      //get
		public String get(String thekey) throws RemoteException;
      //delete 
	     public void delete(String thekey) throws RemoteException; 
	}

