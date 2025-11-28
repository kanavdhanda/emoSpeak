import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Mic, Square, Activity, Brain, Zap, MessageSquare, Plus, Trash2, RefreshCw } from 'lucide-react';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Multi-Chat State
  // Each chat now holds its own 'brainState'
  const [chats, setChats] = useState([{ 
    id: 'default_user', 
    name: 'Session 1', 
    messages: [],
    brainState: { baseline_energy: 0.5, interaction_count: 0, emotional_memories: [], conversation_history: [] }
  }]);
  const [currentChatId, setCurrentChatId] = useState('default_user');
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const currentChat = chats.find(c => c.id === currentChatId) || chats[0];
  const currentBrainState = currentChat.brainState;

  const createNewChat = () => {
    const newId = `user_${Date.now()}`;
    const newChat = { 
      id: newId, 
      name: `Session ${chats.length + 1}`, 
      messages: [],
      brainState: { baseline_energy: 0.5, interaction_count: 0, emotional_memories: [], conversation_history: [] }
    };
    setChats([...chats, newChat]);
    setCurrentChatId(newId);
  };

  const resetEmotions = () => {
    if (!confirm("Are you sure you want to wipe the emotional memory for this session?")) return;
    
    setChats(prevChats => prevChats.map(chat => {
      if (chat.id === currentChatId) {
        return {
          ...chat,
          brainState: { baseline_energy: 0.5, interaction_count: 0, emotional_memories: [], conversation_history: [] }
        };
      }
      return chat;
    }));
    alert("Emotional state reset locally.");
  };

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
    formData.append('user_id', currentChatId);
    // Send current brain state to backend
    formData.append('brain_state', JSON.stringify(currentChat.brainState));

    try {
      const response = await axios.post('http://localhost:8000/process_audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data;
      
      // Update Conversation History & Brain State for Current Chat
      setChats(prevChats => prevChats.map(chat => {
        if (chat.id === currentChatId) {
          return {
            ...chat,
            messages: [...chat.messages, {
              role: 'user',
              text: data.user_input.text,
              emotion: data.user_input.detected_emotion,
              adjusted: data.user_input.adjusted_emotion
            }, {
              role: 'ai',
              text: data.ai_response.text,
              emotion: data.ai_response.emotion,
              percentage: data.ai_response.percentage
            }],
            brainState: data.brain_state // Update local state with backend response
          };
        }
        return chat;
      }));

    } catch (error) {
      console.error("Error processing audio:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex text-white overflow-hidden font-mono">
      
      {/* Sidebar */}
      <div className="w-64 bg-surface border-r border-white/10 flex flex-col p-4 gap-4 z-20">
        <div className="flex items-center gap-2 mb-8">
          <Brain className="w-6 h-6 text-primary" />
          <h1 className="text-xl font-bold tracking-widest">EMOTTS</h1>
        </div>
        
        <button 
          onClick={createNewChat}
          className="flex items-center gap-2 border border-white/20 hover:bg-white/5 text-white p-3 transition-all text-sm uppercase tracking-wider"
        >
          <Plus className="w-4 h-4" />
          New Session
        </button>

        <div className="flex-1 overflow-y-auto space-y-1">
          {chats.map(chat => (
            <button
              key={chat.id}
              onClick={() => setCurrentChatId(chat.id)}
              className={`w-full text-left p-3 transition-all text-sm ${
                currentChatId === chat.id 
                  ? 'bg-primary/20 text-primary border-l-2 border-primary' 
                  : 'text-white/50 hover:text-white'
              }`}
            >
              {chat.name}
            </button>
          ))}
        </div>

        <div className="pt-4 border-t border-white/10">
           <button 
            onClick={resetEmotions}
            className="flex items-center gap-2 text-white/50 hover:text-red-400 text-xs uppercase tracking-widest w-full p-2 transition-all"
          >
            <RefreshCw className="w-3 h-3" />
            Reset Memory
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative">
        
        <div className="flex-1 p-8 grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-7xl mx-auto w-full h-full">
          
          {/* Left Panel: Brain State */}
          <div className="border border-white/10 p-6 flex flex-col gap-6 h-fit bg-surface">
            <div className="flex items-center gap-3 text-primary mb-2">
              <Activity className="w-5 h-5" />
              <h2 className="text-sm font-bold uppercase tracking-widest">Neural State</h2>
            </div>
            
            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-xs text-white/50 mb-2 uppercase">
                  <span>Baseline Energy</span>
                  <span>{(currentBrainState.baseline_energy * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-white/5 h-1">
                  <div 
                    className="h-full bg-primary transition-all duration-1000"
                    style={{ width: `${currentBrainState.baseline_energy * 100}%` }}
                  />
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-white/10 pt-4">
                <div className="flex items-center gap-3">
                  <Zap className="w-4 h-4 text-white/50" />
                  <span className="text-xs text-white/50 uppercase">Interactions</span>
                </div>
                <span className="text-xl font-bold text-primary">{currentBrainState.interaction_count}</span>
              </div>
            </div>
          </div>

          {/* Center Panel: Interaction */}
          <div className="flex flex-col items-center justify-center gap-12">
            
            {/* Visualizer / Button */}
            <div className="relative group">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`relative w-32 h-32 flex items-center justify-center border transition-all duration-300 ${
                  isRecording 
                    ? 'border-primary bg-primary/10' 
                    : 'border-white/20 hover:border-primary hover:bg-white/5'
                }`}
              >
                {isProcessing ? (
                  <div className="animate-spin h-8 w-8 border-t-2 border-b-2 border-primary"></div>
                ) : isRecording ? (
                  <Square className="w-8 h-8 text-primary fill-current" />
                ) : (
                  <Mic className="w-8 h-8 text-white/80" />
                )}
              </button>
            </div>

            <div className="text-center h-8">
              {isRecording && <span className="text-primary animate-pulse text-xs uppercase tracking-[0.2em]">Recording Input...</span>}
              {isProcessing && <span className="text-white/50 animate-pulse text-xs uppercase tracking-[0.2em]">Processing...</span>}
              {!isRecording && !isProcessing && <span className="text-white/30 text-xs uppercase tracking-[0.2em]">Ready</span>}
            </div>

          </div>

          {/* Right Panel: Conversation */}
          <div className="border border-white/10 p-6 h-[600px] overflow-y-auto flex flex-col gap-6 scrollbar-hide bg-surface">
             <div className="flex items-center gap-3 text-primary mb-2 sticky top-0 bg-surface py-2 z-10 border-b border-white/10">
              <MessageSquare className="w-5 h-5" />
              <h2 className="text-sm font-bold uppercase tracking-widest">Log</h2>
            </div>

            {currentChat.messages.length === 0 && (
              <div className="text-white/20 text-center mt-20 text-xs uppercase tracking-widest">
                No Data
              </div>
            )}

            {currentChat.messages.map((msg, idx) => (
              <div key={idx} className={`flex flex-col gap-2 ${msg.role === 'ai' ? 'items-start' : 'items-end'}`}>
                
                {/* Message Bubble */}
                <div className={`max-w-[90%] p-4 border ${
                  msg.role === 'ai' 
                    ? 'border-white/10 bg-white/5 text-white/90' 
                    : 'border-primary/30 bg-primary/5 text-primary'
                }`}>
                  <p className="text-sm leading-relaxed">{msg.text}</p>
                </div>

                {/* Metadata */}
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider opacity-60">
                  {msg.role === 'user' ? (
                    <>
                      <span className="text-primary">{msg.emotion}</span>
                      {msg.adjusted !== msg.emotion && (
                        <span className="text-white/40">→ {msg.adjusted}</span>
                      )}
                    </>
                  ) : (
                    <div className="flex flex-wrap gap-3 max-w-[250px]">
                      {msg.emotion && msg.emotion.map((emo, i) => (
                        <span key={i} className="flex gap-1">
                          <span className="text-white/70">{emo}</span>
                          <span className="text-white/30">{msg.percentage[i]}</span>
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
    </div>
  );
}

export default App;
