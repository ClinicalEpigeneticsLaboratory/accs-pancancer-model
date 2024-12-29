FROM r-base:4.4.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y
RUN apt install adduser -y
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
        libfontconfig1-dev libcurl4-openssl-dev libharfbuzz-dev libfribidi-dev \
        libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        wget libbz2-dev libssl-dev libxml2-dev pandoc -y

# Install Python
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
RUN tar -xvf Python-3.10.0.tgz
RUN cd Python-3.10.0 && ./configure --enable-optimizations && make -j 4 && make altinstall

# Copy python requirements into the container
COPY requirements.txt .

# Install Python dependencies
RUN python3.10 -m pip install -r requirements.txt

# Prepare dir for sesame cache
ENV EXPERIMENT_HUB_CACHE="/usr/local/cache/.ExperimentHub"
RUN mkdir -p $EXPERIMENT_HUB_CACHE

# Copy R requirements into the container
# Install R components
COPY requirements.R sesame_cache.R ./

RUN Rscript requirements.R
RUN Rscript sesame_cache.R

# Download ref data for conumee
WORKDIR /workdir/
RUN wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112618/suppl/GSE112618_RAW.tar
RUN tar -xf GSE112618_RAW.tar

# ENTRYPOINT
CMD ["sh"]
