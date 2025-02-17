FROM ubuntu:oracular

# Suppress debconf warnings
ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN echo 'deb http://archive.ubuntu.com/ubuntu oracular  main universe multiverse' >> /etc/apt/sources.list && \
apt-get update -y && \
apt-get --no-install-recommends -y install \
g++ libtinyxml2-dev

WORKDIR /CANDY_PICKER

# Copy the source code
COPY candy_picker.cpp /CANDY_PICKER/

# Compile the source code
RUN g++ -fopenmp -o candy_picker candy_picker.cpp -ltinyxml2

#add to PATH
ENV PATH="/CANDY_PICKER:${PATH}"

WORKDIR /


CMD ["bash"]
