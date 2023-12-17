from gtts import gTTS

text='no object detected'

file=gTTS(text=text,lang='en')

file.save("D:\a\Project\CSP\no_obj.mp3")

