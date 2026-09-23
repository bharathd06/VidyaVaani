from deep_translator import GoogleTranslator

text_to_translate = """
this is a serious mistake a whole generations of people have done that if you study how should you study hard you must study hard if you work how should you work you must work hard why didn't they tell you you must study joyfully why didn't I tell you you must work lovingly no you must do everything hot and then you complain you complain about everything in life because you're doing everything hot very substation Medical and scientific evidence to show you only when you are in pleasant states of experience does your body and your brain work at their best is that important for you to perform any activity in life well that your body and your brains are working well hello is it important
"""

print("\n--- Translating with Google Translate ---")
translation = GoogleTranslator(source='en', target='kn').translate(text_to_translate)
print(translation)
print("-----------------------------------------")
