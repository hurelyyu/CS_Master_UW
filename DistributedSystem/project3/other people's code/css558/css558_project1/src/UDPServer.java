/**
 * Created by binrusas on 4/8/14.
 */
import java.io.*;
import java.net.*;
import java.util.HashMap;
import java.util.Map;

public class UDPServer{
    private int port;
    private Logger logger ;
    DatagramSocket serverSocket;
    private Map<String,String> kvalues;
    public UDPServer(int port) throws IOException {
        this.port = port;
        serverSocket = new DatagramSocket(port);
        kvalues = new HashMap<String, String>();
        logger = new Logger("UDP-server-log.txt");
        logger.log("Server start running on : " + Inet4Address.getLocalHost());
        System.out.println("Server start running on : " + Inet4Address.getLocalHost());
    }

    private String serveRequest(String request){
        String [] tokens = request.split(",");
        String command = tokens[0];
        String value;
        if(command.equals("GET")){
            value =  kvalues.get(tokens[1]);
            if(value != null){
                return "The value of key (" + tokens[1] + ") is (" + value +")";
            }
        }
        else if(command.equals("PUT")){
            kvalues.put(tokens[1], tokens[2]);
            if(kvalues.containsKey(tokens[1])){
                return "Server has succeeded to handle request = (" + request + ")";
            }
            else{
                return "Server has failed to handle request =  (" + request + ")";
            }
        }
        else if(command.equals("DELETE")){
            kvalues.remove(tokens[1]);
            if(!kvalues.containsKey(tokens[1])){
                return "Server has succeeded to handle request =  (" + request + ")";
            }
            else{
                return "Server has failed to handle request = (" + request + ")";
            }
        }
        return "Server has failed to handle request = (" + request + ")";
    }

    public void listenAndServeRequests() throws IOException {

        DatagramPacket receivedPacket , sendPacket;
        while(true)
        {
            byte[] receiveData = new byte[256];
            receivedPacket = new DatagramPacket(receiveData, receiveData.length);
            serverSocket.receive(receivedPacket);
            String request = getRidOfAnnoyingChar(receivedPacket);
            logger.log("Server has received request from client: " + receivedPacket.getAddress() +
                    "\nThe Request is : "+ request);
            System.out.println("Server has received request from client: " + receivedPacket.getAddress() +
                    "\nThe Request is : "+ request);

            String result = serveRequest(request);

            InetAddress clientIPAddress = receivedPacket.getAddress();
            int clientPort = receivedPacket.getPort();

            byte[] sendData = result.getBytes();
            sendPacket = new DatagramPacket(sendData, sendData.length, clientIPAddress, clientPort);
            serverSocket.send(sendPacket);
            logger.log("Server has replied to client IP Address : " + clientIPAddress + "," +
                    " port: " + clientPort + "\nThe reply is : (" + result + ")");
            System.out.println("Server has replied to client IP Address : " + clientIPAddress + "," +
                    " port: " + clientPort + "\nThe reply is : (" + result + ")");
        }
    }

    public void closeLogger(){
        logger.close();
    }

    public String getRidOfAnnoyingChar(DatagramPacket packet){
        String result = new String(packet.getData());
        char[] annoyingchar = new char[1];
        char[] charresult = result.toCharArray();
        result = "";
        for(int i=0;i<charresult.length;i++){
            if(charresult[i]==annoyingchar[0]){
                break;
            }
            result+=charresult[i];
        }
        return result;
    }

    public static void main(String args[]){
        if(args.length < 1){
            System.out.println("Error: port must be provided");
            System.exit(2);
        }
        UDPServer server = null ;
        try{
            int port = Integer.parseInt(args[0]);
            server = new UDPServer(port);
            server.listenAndServeRequests();
        }catch (Exception ex){
            ex.printStackTrace();
            server.closeLogger();
        }
    }
}
