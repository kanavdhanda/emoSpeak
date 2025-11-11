# Emo-TTS

Emo-TTS is an ensemble-based system designed to generate emotionally aligned speech responses.  
It is divided into three sections, each with a specific purpose, strengths, and limitations.

1. **Emotion & Sentiment Understanding**  
2. **Emotion-Conditioned Response Generation**  
3. **Emotionally Expressive Speech Synthesis**

---

## Section 1: Emotion & Sentiment Detection

### What This Section Does
This section identifies the user's emotional state from multimodal input (speech, text, potentially facial expressions).  
It provides the emotional context that influences how the system responds.

### Why This Section Exists
Machines responding with appropriate emotional tone require accurate understanding of the user’s current mood.  
If the system misunderstands the user’s emotional state, everything downstream becomes inconsistent or awkward.

### How This Section Works
1. Speech → Text using an ASR model such as Whisper.  
2. Text → Embeddings using BERT-based models.  
3. Embeddings → Sentiment/Emotion prediction using:
   - K-means clustering  
   - K-NN  
   - Neural Networks  
   - LLM-based classifiers  
4. Dataset We Used for trainiing
    - IMDB
    - Twitter
These models capture semantic sentiment but currently *lack multimodal depth* (tonality, prosody, facial expression). However they helped us create a baseline for evaluation.


### Limitations / Challenges to the above mentioned 
- Whisper and similar models don’t inherently capture tone.
- Emotion detection from text alone is shallow.
- Accuracy varies heavily depending on phrasing.
- Multimodal integration (audio + text + facial cues) is still not implemented.
- No personalization yet, so the system treats every user the same.

### What we plan to fix

#### Personalization
To address long-term adaptation, future development aims to simulate functional analogies to:
- **Neuroplasticity**: adjusting emotional interpretation over time based on repeated user interactions  
- **Neurogenesis**: allowing new “patterns” of emotional mapping to form beyond initial training  

We aim to mimic the human brain for introducting personalization and Are using these methods to achieve the task:

#### Multimodal capabilites

---

## Section 2: Emotion-Aligned Response Generation

### What This Section Does
This section generates:
- The textual reply  
- The emotional distribution that the response should carry

### Why This Section Exists
Understanding emotion is not enough.  
The system needs to:
- Respond meaningfully  
- Keep the emotional tone aligned with the user's state  
- Produce a coherent emotional directive for speech generation  

### How This Section Works
Uses an LLM (local or external) to:
- Interpret the user’s query  
- Combine detected emotion with conversational intent  
- Produce a response and emotion map  

Possible LLMs:
- Ollama-hosted models  
- Remote APIs (Gemini, etc.)  
- Any reasonably capable LLM with emotional conditioning  

### Limitations / Challenges
- Emotional conditioning via prompt engineering is unreliable at times.
- LLMs can misjudge emotional tone, producing mismatched responses.
- Needs guardrails to avoid overly dramatic or flat emotional output.
- Personalization logic is not implemented; output feels generic.

---

## Section 3: Emotionally Expressive Speech Synthesis

### What This Section Does
This section converts the textual output + emotion distribution into expressive, human-like audio.

### Why This Section Exists
Emotionally expressive speech is the final user-facing output.  
Without it, the entire system would sound monotone, robotic, or emotionally tone-deaf.

### How This Section Works
Krrish is developing the synthesis pipeline.  
This section takes a JSON-like structure such as:

```json
{
  "text": "Hello, I am happy today",
  "emotion": ["Cheerful", "Warm"],
  "percentage": ["77%", "23%"]
}
```

The system uses:
- Emotion-weighted voice samples
- A pretrained TTS model [CosyVoice V3](https://funaudiollm.github.io/cosyvoice3/)
- A dataset with emotional annotations (Emilia dataset)

Multiple different models both with this flow and pretrained were tested, (code mentioned in repo), however the suggested flow gave us better results(tested on human Perception).

---

# Merging Everything Together
All three sections are connected for a demo, and frontend and backend codes for the same are also provided in this repository.