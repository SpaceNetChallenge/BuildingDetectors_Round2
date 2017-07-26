FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER kohei

# Install dependent packages via apt-get
RUN apt-get -y update &&\
    echo ">>>>> packages for building python" &&\
    apt-get --no-install-recommends -y install \
      g++ \
      libsqlite3-dev \
      libssl-dev \
      libreadline-dev \
      libncurses5-dev \
      lzma-dev \
      liblzma-dev \
      libbz2-dev \
      libz-dev \
      libgdbm-dev \
      build-essential \
      cmake \
      make \
      wget \
      unzip \
      &&\
    echo ">>>>> packages for building python packages" &&\
    apt-get --no-install-recommends -y install \
      libblas-dev \
      liblapack-dev \
      libpng-dev \
      libfreetype6-dev \
      pkg-config \
      ca-certificates \
      libhdf5-serial-dev \
      postgresql \
      libpq-dev \
      curl \
      &&\
    apt-get clean

# -=-=-=- Java -=-=-=-
RUN apt-get --no-install-recommends -y install software-properties-common &&\
    add-apt-repository ppa:webupd8team/java -y &&\
    apt-get update &&\
    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections &&\
    apt-get install -y oracle-java8-installer &&\
    apt-get clean

# -=-=-=- Anaconda -=-=-=-
RUN ANACONDA_URL="https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh" &&\
    ANACONDA_FILE="anaconda.sh" &&\
    mkdir -p /opt &&\
    cd /opt &&\
    wget -q --no-check-certificate $ANACONDA_URL -O $ANACONDA_FILE &&\
    echo "4447b93d2c779201e5fb50cfc45de0ec96c3804e7ad0fe201ab6b99f73e90302  ${ANACONDA_FILE}" | sha256sum -c - &&\
    bash $ANACONDA_FILE -b -p /opt/conda &&\
    rm $ANACONDA_FILE
ENV PATH "$PATH:/opt/conda/bin"

# -=-=-=- Python packages (py35 env) -=-=-=-
COPY py35.yml /opt/
RUN cd /opt &&\
    conda env create -f py35.yml

# Keras
RUN mkdir /root/.keras
COPY keras.json /root/.keras/

# Deploy OSMdata
RUN mkdir /root/osmdata
COPY osmdata /root/osmdata/
RUN unzip /root/osmdata/las-vegas_nevada.imposm-shapefiles.zip \
        -d /root/osmdata/las-vegas_nevada_osm/ &&\
    unzip /root/osmdata/shanghai_china.imposm-shapefiles.zip \
        -d /root/osmdata/shanghai_china_osm/ &&\
    unzip /root/osmdata/paris_france.imposm-shapefiles.zip \
        -d /root/osmdata/paris_france_osm/ &&\
    unzip /root/osmdata/ex_s2cCo6gpCXAvihWVygCAfSjNVksnQ.imposm-shapefiles.zip \
        -d /root/osmdata/ex_s2cCo6gpCXAvihWVygCAfSjNVksnQ_osm/

# Copy and unzip visualizer
COPY code/visualizer-2.0.zip /root/
RUN unzip -d /root/visualizer-2.0 /root/visualizer-2.0.zip

# Deploy codes
COPY code /root/
RUN chmod a+x /root/train.sh &&\
    chmod a+x /root/test.sh

ENV PATH $PATH:/root/

# Env
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
WORKDIR /root/
