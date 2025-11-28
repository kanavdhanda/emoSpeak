import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Mic, Square, Activity, Brain, Zap, MessageSquare } from 'lucide-react';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [brainState, setBrainState] = useState({ baseline_energy: 0.5, interaction_count: 0 });
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = sendAudio;
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

  const sendAudio = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    try {
      const response = await axios.post('http://localhost:8000/process_audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data;
      
      // Update Conversation History
      setConversation(prev => [...prev, {
        role: 'user',
        text: data.user_input.text,
        emotion: data.user_input.detected_emotion,
        adjusted: data.user_input.adjusted_emotion
      }, {
        role: 'ai',
        text: data.ai_response.text,
        emotion: data.ai_response.emotion,
        percentage: data.ai_response.percentage
      }]);

      // Update Brain State
      setBrainState(data.brain_state);

    } catch (error) {
      console.error("Error processing audio:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-8 flex flex-col items-center justify-center relative overflow-hidden">
      {/* Background Gradients */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse-slow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1.5s' }}></div>
      </div>

      {/* Header */}
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-bold mb-2 tracking-tighter gradient-text">NEURO-ADAPTIVE</h1>
        <p className="text-white/50 tracking-widest text-sm uppercase">Synthetic Emotional Intelligence</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full max-w-6xl">
        
        {/* Left Panel: Brain State */}
        <div className="glass-panel p-6 flex flex-col gap-6 h-fit">
          <div className="flex items-center gap-3 text-primary mb-2">
            <Brain className="w-6 h-6" />
            <h2 className="text-xl font-semibold">Amygdala State</h2>
          </div>
          
          <div className="space-y-4">
            <div className="bg-white/5 p-4 rounded-xl">
              <div className="flex justify-between text-sm text-white/70 mb-1">
                <span>Baseline Energy</span>
                <span>{(brainState.baseline_energy * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-1000"
                  style={{ width: `${brainState.baseline_energy * 100}%` }}
                />
              </div>
            </div>

            <div className="bg-white/5 p-4 rounded-xl flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Activity className="w-5 h-5 text-secondary" />
                <span className="text-sm text-white/70">Interactions</span>
              </div>
              <span className="text-2xl font-mono font-bold">{brainState.interaction_count}</span>
            </div>
          </div>
        </div>

        {/* Center Panel: Interaction */}
        <div className="flex flex-col items-center justify-center gap-8">
          
          {/* Visualizer / Button */}
          <div className="relative group">
            <div className={`absolute inset-0 bg-gradient-to-r from-primary to-accent rounded-full blur-2xl transition-opacity duration-500 ${isRecording ? 'opacity-50' : 'opacity-0'}`}></div>
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
              className={`relative w-40 h-40 rounded-full flex items-center justify-center border-4 transition-all duration-300 ${
                isRecording 
                  ? 'border-accent bg-accent/10 scale-110' 
                  : 'border-white/10 bg-white/5 hover:border-primary/50 hover:bg-white/10'
              }`}
            >
              {isProcessing ? (
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
              ) : isRecording ? (
                <Square className="w-12 h-12 text-accent fill-current" />
              ) : (
                <Mic className="w-12 h-12 text-white/80" />
              )}
            </button>
          </div>

          <div className="text-center h-8">
            {isRecording && <span className="text-accent animate-pulse font-mono tracking-widest">RECORDING AUDIO DATA...</span>}
            {isProcessing && <span className="text-primary animate-pulse font-mono tracking-widest">PROCESSING NEURAL PATHWAYS...</span>}
            {!isRecording && !isProcessing && <span className="text-white/30 font-mono text-sm">TAP TO INTERFACE</span>}
          </div>

        </div>

        {/* Right Panel: Conversation */}
        <div className="glass-panel p-6 h-[600px] overflow-y-auto flex flex-col gap-4 scrollbar-hide">
           <div className="flex items-center gap-3 text-secondary mb-2 sticky top-0 bg-surface/95 backdrop-blur-xl py-2 z-10">
            <MessageSquare className="w-6 h-6" />
            <h2 className="text-xl font-semibold">Live Feed</h2>
          </div>

          {conversation.length === 0 && (
            <div className="text-white/20 text-center mt-20 italic">
              No neural activity detected.
            </div>
          )}

          {conversation.map((msg, idx) => (
            <div key={idx} className={`flex flex-col gap-2 ${msg.role === 'ai' ? 'items-start' : 'items-end'}`}>
              
              {/* Message Bubble */}
              <div className={`max-w-[90%] p-4 rounded-2xl ${
                msg.role === 'ai' 
                  ? 'bg-gradient-to-br from-white/10 to-white/5 rounded-tl-none border border-white/10' 
                  : 'bg-primary/20 rounded-tr-none border border-primary/20'
              }`}>
                <p className="text-lg leading-relaxed">{msg.text}</p>
              </div>

              {/* Metadata */}
              <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider opacity-60">
                {msg.role === 'user' ? (
                  <>
                    <span className="text-primary">{msg.emotion}</span>
                    {msg.adjusted !== msg.emotion && (
                      <span className="text-white/40">→ {msg.adjusted}</span>
                    )}
                  </>
                ) : (
                  <div className="flex gap-3">
                    {msg.emotion && msg.emotion.map((emo, i) => (
                      <span key={i} className="flex gap-1">
                        <span className="text-secondary">{emo}</span>
                        <span className="text-white/50">{msg.percentage[i]}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}

export default App;
