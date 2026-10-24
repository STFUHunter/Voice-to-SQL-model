import speech_recognition as sr
import sys
import os 
from ctypes import *

#init the speech reconizer class
r = sr.Recognizer()

#since the recognizer outputs all the error messages after checking all of my ports to the console, I have to redirect all that somewhere else (a black hole)
ERROR_HANDLE_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass 

c_error_handler = ERROR_HANDLE_FUNC(py_error_handler)

try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except OSError:
    pass 

def record_speech():
    while(True):
        try:
            #access to the microphone
            with sr.Microphone() as source:
                #prepare the recognizer for ambient audio
                r.adjust_for_ambient_noise(source, duration=1.0)
                
                #listen to the user's audio
                audio = r.listen(source, phrase_time_limit=30)

                #convert audio to a string using google
                text = r.recognize_whisper(audio, model="small")

                #if the audio was recognized it will be returned
                return text
        
        #raising an exception if audio input is not recognized or the audio itself is not reconized 
        except sr.RequestError as e:
            print(f"Sorry, we can't request results: {0}.".format(e))
        
        except sr.UnknownValueError as e:
            print(f"Sorry, unknown error occured")

while(True):
    speech = record_speech()
    print(speech)