This instruction is for Mac and Linux users.
We have executed makefile and generated .class files for you. But you can also make .class files by yourself. 

————————————————————————————————————
How to open a server using command line
————————————————————————————————————

1. Access the storage directory of .class files
Command:
cd /(project path)/src

2.1 Open a TCP server
Command:
Java TCPServerMain <portNumber>

2.2 Open a UDP server
Command:
Java UDPServerMain <portNumber>

3. Q & A
Q: What is (project path)?
A: The path of the project folder.
Q: What is <portNumber>?
A: The number of port you want the server to be opened on.

————————————————————————————————————
How to open a client using command line
————————————————————————————————————

1. Access the storage directory of .class files
Command:
cd /(project path)/src

2.1. Open a TCP client
Command:
Java TCPClientMain <operation><server ip address or hostname><server portNumber>

2.2. Open a UDP client
Command:
Java UDPClientMain <operation><server ip address or hostname><server portNumber>

3. Q & A
Q: What is (project path)?
A: The path of the project folder.
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

1. Locate testPairs.txt(our command set script) in the same directory of testMain.class

2. Access the storage directory of .class files
Command:
cd /(project path)/src

3. Open a TCP or UDP server. This step is described above.
The default <portNumber> for the server is 1234. 
You can also change the port number in testPairs.txt and use the same port number to open your server.

4.1. Open a TCP client to read commands from testPairs.txt
Command:
Java testMain TCP

4.2. Open a UDP client to read commands from testPairs.txt
Command:
Java testMain UDP

5. Q & A
Q: What is (project path)?
A: The path of the project folder.
Q: How to quickly change the port number in testPairs.txt? 
A: Find/Replace 1234 to the port number you want.
Q: How to change the test commands in testPairs.txt? 
A: Follow the arguments instruction in part "How to open a client using command line", you'll be fine.