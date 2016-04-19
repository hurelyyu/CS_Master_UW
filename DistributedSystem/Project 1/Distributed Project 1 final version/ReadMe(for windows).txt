This instruction is for Windows users.

————————————————————————————————————
How to open a server using command line
————————————————————————————————————

1.1 Open a TCP server
Command:
Java -cp ./(path)/558.jar TCPServerMain <portNumber>

1.2 Open a UDP server
Command:
Java -cp ./(path)/558.jar UDPServerMain <portNumber>

2. Q & A
Q: What is (path)?
A: The relative path of the 558.jar file.
Q: What is <portNumber>?
A: The number of port you want the server to be opened on.

————————————————————————————————————
How to open a client using command line
————————————————————————————————————

1.1. Open a TCP client
Command:
Java -cp ./(path)/558.jar TCPClientMain <operation><server ip address or hostname><server portNumber>

1.2. Open a UDP client
Command:
Java -cp ./(path)/558.jar UDPClientMain <operation><server ip address or hostname><server portNumber>

2. Q & A
Q: What is (path)?
A: The relative path of the 558.jar file.
Q: What is <operation>?
A: The operation you want to execute on the server. 
   For now the client accepts 3 kind of operations: 
   PUT/<key>/<value>, GET/<key>, DELETE/<key>
Q: What is <server ip address or hostname>?
A: The ip address or hostname of your server.
   For now just use "localhost".
Q: What is <server portNumber>?
A: The number of port your server opened on.

————————————————————————————————————————————————
How to make a client read commands from a script
————————————————————————————————————————————————

1. Locate testPairs.txt(our command set script) in the same directory of 558.jar

2. Open a TCP or UDP server. This step is described above.
The default <portNumber> for the server is 1234. 
You can also change the port number in testPairs.txt and use the same port number to open your server.

3.1. Open a TCP client to read commands from testPairs.txt
Command:
Java -cp ./(path)/558.jar testMain TCP

3.2. Open a UDP client to read commands from testPairs.txt
Command:
Java -cp ./(path)/558.jar testMain UDP

4. Q & A
Q: What is (path)?
A: The relative path of the 558.jar file.
Q: How to quickly change the port number in testPairs.txt? 
A: Find/Replace 1234 to the port number you want.
Q: How to change the test commands in testPairs.txt? 
A: Follow the arguments instruction in part "How to open a client using command line", you'll be fine.