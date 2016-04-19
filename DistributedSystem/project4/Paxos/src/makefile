JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	Acceptor.java \
	ClientMain.java \
	KVProcessor.java \
	LogProcessor.java \
	myLog.java \
	Paxos.java \
	Proposer.java \
	RMIClient.java \
	RMIinterface.java \
	RMIServer.java \
	RMIServerStoreData.java \

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
