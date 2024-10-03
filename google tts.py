# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:47:59 2022

@author: USER
"""

from gtts import gTTS
from playsound import playsound

import os

mytext = 'നിങ്ങൾ മയക്കത്തിലാണ്. അൽപ്പം വിശ്രമിക്കാനോ നടക്കാൻ പോകാനോ ഞാൻ ശുപാർശ ചെയ്യുന്നു.'
language = 'ml'

myobj = gTTS(text=mytext, tld = "co.in",  lang=language, slow=False)
myobj.save("drowsy.mp3")

playsound("drowsy.mp3")

