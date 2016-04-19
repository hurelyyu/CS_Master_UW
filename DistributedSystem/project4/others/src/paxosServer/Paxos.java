package paxosServer;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface Paxos extends Remote {
	public void prepare() throws RemoteException;
	
	public void accept() throws RemoteException;
	
	public void commit() throws RemoteException;
	
}
