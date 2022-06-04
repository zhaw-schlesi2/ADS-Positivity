FROM python:3

# Maindirectory is positivity - please keep all necessary files within this folder
ADD positivity /positivity

# Install all requirements
RUN pip install -r ./positivity/requirements.txt

# Need to be done in docker, otherwise an error occurs
# NOT needed to use application otherwise
RUN pip install --upgrade protobuf==3.20.0

# Expose the 8888 tcp ip
EXPOSE 8888/tcp

# Starts the server with a pretrained model
CMD [ "python", "./positivity/positivity.py", "run", "-m",  "./positivity/pretrained-models/bertlstmOPT.pth.001" ]

