JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	ClassCaseException.java \
	clientlog.java \
	HashmapYu.java \
	ImplementHashmapYu.java \
	serverlog.java \
	TCPclient.java \
	TCPserver.java \

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
