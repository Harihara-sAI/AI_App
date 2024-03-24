The code was written after referencing multiple sources on the internet, the main reference being a YouTube video by Prompt Engineering (Link: https://www.youtube.com/watch?v=RIWbalZ7sTo).
Please check the "requirements.txt" file
The application is hosted on streamlit, thus to run it, open the terminal and use the "streamlit run app.py" command to run it.

Ideas for improvement:
1. The code can be made more modular by adding functions for tasks such as converting PDF to text and converting that text to chunks.
2. The hosting platform can be changed as the streamlit interface runs a bit slower.
3. The app currently runs on OpenAI's free plan, thus it asks for billing details after a certain number of uses. Can be mitigated by using getting the paid subscription's key. This can be done by changing the code in line 13.