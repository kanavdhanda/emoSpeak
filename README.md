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


---
## Section 3: Emotionally Expressive Speech Synthesis

### What This Section Does
This section converts the output from Section 2 (text + emotional distribution) into expressive, human-like audio. It ensures that the final speech reflects the intended emotional blend and aligns with the user’s perceived mood.

### Why This Section Exists
Even with accurate emotion detection and strong response generation, flat or mismatched audio breaks immersion.  
This section produces the final emotional layer, making the system feel natural, consistent, and context-aware.

### How This Section Works
The synthesis pipeline takes structured emotional directives such as:

```json
{
  "text": "Hello, I am happy today",
  "emotion": ["Cheerful", "Warm"],
  "percentage": ["77%", "23%"]
}
```

### The Processing Flow Includes

#### 1. Emotional Interpretation
Convert the emotion percentages into a weighted emotional representation that defines how strongly each emotion influences the final output.

#### 2. Emotion-Specific Audio Extraction
Identify suitable emotional voice samples from the **Emilia dataset**, matched to the target emotional distribution.

#### 3. Zero-Shot In-Context Generation
Feed these selected samples into **CosyVoice V3**, enabling the system to:
- Extract emotional prosody  
- Transfer vocal style  
- Blend multiple emotional cues  
- Apply these characteristics to the generated text  

#### 4. Final Audio Output
Produce expressive speech by combining:
- The textual response generated in Section 2  
- The emotional blend defined by the input distribution  


### Why This Approach Should Work

CosyVoice V3 enables:
- Zero-shot emotional style transfer  
- High-quality natural speech synthesis  
- Smooth interpolation between emotion types  
- Strong performance without fine-tuning  

### Experiments Conducted

Several configurations were explored:
- Pretrained TTS without emotional conditioning  
- Alternative emotional datasets  
- Different numbers of reference samples  
- Varying emotional blending strategies  

Human perception tests were based on:
- Clearer emotional expression  
- More natural-sounding prosody  
- Improved consistency across different emotions  
<br></br>
---

# Merging Everything Together
All three sections are connected for a demo, and frontend and backend codes for the same are also provided in this repository.