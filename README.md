diaqres
=======


This repository contains the code and setup necessary to reproduce results published in a thesis titled
'Diacritics Restoration for Slovak Texts Using Deep Neural Networks'

Data
----

First, download the relevant data from http://davinci.fmph.uniba.sk/~suppa1/master_thesis_attachment.zip

Dependencies
------------

In order to use the scripts contained in this repository create a new virtual
envrionrment by running

        $ virtualenv venv

then turn this virtualenv on by running

        $ source venv/bin/activate.sh

provided you use the Bash shell.

Training
--------

Since the code makes use of the `sacred` logging framework, we strongly
encourage you to create a directory called `sacred_logs` by running

        $ mkdir sacred_logs

The training scripts can then be ran by executing commands such as for
instance:

        $ python train.py -F sacred_logs with 'filename=../../skwiki_with_diac_train.txt' 'n_layers=1' 'n=51' 'embed_size=100' 'hidden_size=200' 'minibatch_len=500' 'save_each_epochs=1000000' 'print_each_epochs=5000' 'model_type=gru'  'bidirectional=False'

Testing
-------

Once a model is trained, it can be tested by running a script in the following
way:

        $ python test.py test_file model_file minibatch_len


Note that the scripts mentioned in this README are located in the `scripts/`
directory.
