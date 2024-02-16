FROM triggers/base:latest

COPY . /

WORKDIR /
RUN pip install .

WORKDIR /sequence_level_trigger_warning_assignment
