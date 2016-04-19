Yaqun Yu and Xiao Liang 558 Project2 ReadMe
********************
For local two users:
********************

Open Server terminal on one computer:
1. Get IP address
2. go to the project Server folder directory(use cd command)
3. make
4. rmiregistry &
5. java RMIserver 1099(default port number)

open Client terminal on another computer:
1. go to the project Client folder directory(use cd command)
2. make
3. java RMIclient2 <Server IP Address>

and Wala!
the result will show up

************************
For VM on school server:
************************

1. get into 558 project VM n01, use as client
2. put Client files on /home/NetID/Client/
3. open another VM n02, use as server. 
4. put Server files on /home/NetID/Server/ through VM n01
5. repeat local instruction for each side.
6. command to check file: vi <File full name, ex: RMIServerLog.txt>