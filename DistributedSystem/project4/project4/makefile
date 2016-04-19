JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	RMIServerInterfaceImpl.java \
	RMIServerInterface.java \
	UtilityClasses.java \
	RMIClient.java \
	RMIServer.java \
	Paxos.java \
	PaxosInterface.java \

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
