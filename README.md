# Positivity

A python utility to classify reviews into positive or negative with the help of machine learning and IMDB reviews for training

### Hardware requirements

An nvidia GPU with functional CUDA drivers installed is recommended. Alternatively you may use AMDs ROCm technology. Training on the CPU, while possible is NOT feasible.

### Setup

- Install the requirements:

```
pip3 install -r requirements.txt
```

- Create a dev account on IMDB to fetch training data if you need it [here](https://imdb-api.com/). A basic account is free and will allow 100 requests per day. To spare you that headache A dataset is included in this repository.

- If you get a "too many files open" error, increase the limit. `ulimit -n 10000`. The DataLoader keeps the files open when indexing to read them later.

### About the model

This utility makes use of Googles pre-trained BERT sentence classifier to tokenize and process the reviews before being passed to additional layers used for classification. This classifier is based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). BERT iself is "in theory" multilanguage capable and will create vectors that belong into certain clusters.

Aside from that, a Long Short-Term Memory `LSTM` network is used in conjunction with the output from BERT. The tokens are feed to both and the outputs later merged behind additional layers which yielded improved model performance during tests.


### Training

The `./positivity.py train -h` command should be pretty self explaining. Most values are set by default to the a tested optimum. You will however need to collect the IMDB data beforehand. For this you may use `./positivity.py collect -h`. With this repository comes a list with movie names to get a somewhat balanced dataset. Additionally an already created dataset is included. 

Note that for additional outputs you can set the `DEBUG=y` environment variable. This will provide some real time samples with actual numbers during training.

Furthermore you will find a wrapper `./positivity-tensorboard.sh`. This will start a tensorboard webserver (web interface: `127.0.0.1:6006`) that provides some graphs during training.

### Using the model for positivity raing

The model was trained to guess the judgment based on given text and it will act accordingly. 

To make requests first the server needs to be started. `./positivity.py run -m modelpath`. The server listens on `tcp://127.0.0.1:8888` by default and the used protocol is `0MQ`. Once you have connected to it, simply send the raw text and you will get a `json` response.

For simplicity a client is included and can be used for this purpose. `./positivity.py client -t "text"`

### Limits

The model is limited to a maximum of 512 characters. This limit is imposed by googles sentence classifier (BERT). Longer texts may work but they are internally cut off. 