# Emo-tts

Emo-tts is majorly an ensemble of past models, we have theorised a way to improve upon the current architecture of emotion filled model speech-to-speech

The projects are divided into three different sections:
Section 1: This section evaluates the sentiment of a user, based on either multimodal capabilites/ using setiment
Section 2: This section is responsible for generating a response based on what the user query is and the emotions detected in
           section 1. This section also tells the emotions the responding audio should have.
Section 3: Based on the output of Section 2, the appropriate audio should be generated

Section 1:
Current Sentiment Analysis models are quite good at their tasks with some models acheiving upto 94% accuracy in predicting user sentiments.
We will be using one such model for this task.
The modal we will be using is: 
We have also tested multiple different approaches like
K-means
KNN
Neural Networks
And an llm
all using a bert based encoding model for creating embeddings for the user prompt. This model does not consider the multiple different data types, like tonality in the voice or facial expressions of the user, helping us create a baseline for evaluation.

To this, there is the problem of personalization. I want to use two abilities of the human brain called neurogenesis and neuroplasticity to improve the personalization according to a person. This is also a part of the section 2 of this project to generate apt responses.

(yet to be thought on how to implement this task)

Section 2:
Section 2 would have an appropriate prompt going to a llm/a ollama model/gemini api for the task of this section.


Section 3:
Developed By Krrish(Text to Voice)
Krrish is working on how he can create audio samples that accurately represent emotions
In this current approach
Krrish has to try
Taking in a json parameter, wherein, the data is the text that the model has to reproduce and the emotion with which the model has to say it.

for eg:
{
    "text":"Hello i am happy today",
    "emotion": ["Cheerful","warm"],
    "percentage":[ "77%","23%]

}

So the model should return an audio saying Hello i am happy today in a cheerful tone.

We have modeled the sample input voices around the dataset [https://huggingface.co/datasets/laion/Emilia-with-Emotion-Annotations]{laion/Emilia-with-Emotion-Annotations}
And theorised that picking voices with that percent of those emotions to feed into a pretrained model is an appropriate way
The pretrained model we are using: [https://funaudiollm.github.io/cosyvoice3/]{CosyVoice V3}