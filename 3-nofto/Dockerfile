# To be built on a CPU-based system

FROM ubuntu:16.04
LABEL maintainer dlindenbaum

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        software-properties-common

# Install Java 
RUN add-apt-repository ppa:webupd8team/java -y && \
        apt-get update && \
        echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
        apt-get install -y oracle-java8-installer && \
        apt-get clean

RUN mkdir round2Code

# copy entire directory where docker file is into docker container at /opt/round2Code
COPY . /opt/round2Code/

WORKDIR /opt/round2Code

# create bin folder
RUN mkdir bin

# compile Java Code
RUN javac -sourcepath src -cp bin -d bin src/*.java

