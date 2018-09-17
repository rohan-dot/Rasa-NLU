from rasa_core.channels import HttpInputChannel
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_slack_connector import SlackInput


nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/weathernlu')
agent = Agent.load('./models/dialogue', interpreter = nlu_interpreter)

input_channel = SlackInput('xoxp-417608169942-415715967377-416040286644-cf5288889c4973ce423a4f0a57fd104e', #app verification token
							'xoxb-417608169942-415719442849-WsH0mwJurOZlYmpsP9VdEl97', # bot verification token
							'gYwQh7fsFJDHNJIqJ6ztYyUm', # slack verification token
							True)

agent.handle_channel(HttpInputChannel(5004, '/', input_channel))
