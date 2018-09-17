How to use this module

1. pip install -r requirements.txt
(this will install rasa_nlu and rasa_core)

2. install npm (node.js  from official site)
3. pip install rasa_nlu[spacy]
4  python -m spacy download en (Download spacy for the english language)

After all the necessary libraries have been installed

Run nlu_model.py on cmd
to make changes/ see the training data go to data folder using cd data
and type rasa-nlu-trainer here you can see all the intents and entities
to train model run train_init.py from cmd
and to have test conversations run train_online.py from cmd

after you have had a conversation save it as  stories.md and replace the folder in data called stories.md by this new stories.md
