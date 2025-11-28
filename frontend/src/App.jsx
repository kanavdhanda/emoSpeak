import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Mic, Square, Activity, Brain, Zap, MessageSquare, Plus, Trash2, RefreshCw, Send } from 'lucide-react';
import ParticleOrb from './ParticleOrb';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Multi-Chat State
  const [chats, setChats] = useState([{ 
    id: 'default_user', 
    name: 'Session 1', 
    messages: [],
    brainState: { baseline_energy: 0.5, interaction_count: 0, emotional_memories: [], conversation_history: [] }
  }]);
  const [currentChatId, setCurrentChatId] = useState('default_user');
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const chatEndRef = useRef(null);

  const currentChat = chats.find(c => c.id === currentChatId) || chats[0];
  const currentBrainState = currentChat.brainState;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentChat.messages]);

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
    formData.append('brain_state', JSON.stringify(currentChat.brainState));

    try {
      const response = await axios.post('http://localhost:8000/process_audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data;
      
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
            brainState: data.brain_state
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
    <div className="min-h-screen bg-background flex text-white overflow-hidden font-mono selection:bg-primary selection:text-black">
      
      {/* Sidebar */}
      <div className="w-64 bg-surface/50 flex flex-col p-6 gap-6 z-20 hidden lg:flex">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
             <Brain className="w-5 h-5 text-black" />
          </div>
          <h1 className="text-xl font-bold tracking-widest">EMOTTS</h1>
        </div>
        
        <button 
          onClick={createNewChat}
          className="flex items-center gap-3 text-white/70 hover:text-white p-3 rounded-lg hover:bg-white/5 transition-all text-sm uppercase tracking-wider border border-transparent hover:border-white/10"
        >
          <Plus className="w-4 h-4" />
          New Session
        </button>

        <div className="flex-1 overflow-y-auto space-y-1 pr-2">
          {chats.map(chat => (
            <button
              key={chat.id}
              onClick={() => setCurrentChatId(chat.id)}
              className={`w-full text-left p-3 rounded-lg transition-all text-sm ${
                currentChatId === chat.id 
                  ? 'bg-white/10 text-white font-bold' 
                  : 'text-white/40 hover:text-white hover:bg-white/5'
              }`}
            >
              {chat.name}
            </button>
          ))}
        </div>

        <div className="pt-4 border-t border-white/10">
           <button 
            onClick={resetEmotions}
            className="flex items-center gap-2 text-white/40 hover:text-red-400 text-xs uppercase tracking-widest w-full p-2 transition-all"
          >
            <RefreshCw className="w-3 h-3" />
            Reset Memory
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative">
        
        {/* Top Bar: Brain State */}
        <div className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-background/50 backdrop-blur-sm z-10">
          {/* <div className="flex items-center gap-6">
             <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-primary" />
                <div className="flex justify-between text-xs text-white/50 mb-2 uppercase">
                  <span>Avg Intensity</span>
                  <span>{(currentBrainState.baseline_energy * 100).toFixed(1)}</span>
                </div>
             </div>
             <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-primary" />
                <span className="text-xs text-white/50 uppercase tracking-widest">Turns</span>
                <span className="text-sm font-bold text-primary">{currentBrainState.interaction_count}</span>
             </div>
          </div> */}
          <div className="text-xs text-white/30 uppercase tracking-widest">
            {isRecording ? "Listening..." : isProcessing ? "Thinking..." : "Idle"}
          </div>
        </div>

        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          
          {/* Center: Visualizer & Controls */}
          <div className="flex-1 flex flex-col items-center justify-center relative p-8">
            
            {/* The Orb */}
            <div className="relative mb-12">
               <ParticleOrb isActive={isRecording || isProcessing} />
               
               {/* Center Button Overlay */}
               <div className="absolute inset-0 flex items-center justify-center">
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isProcessing}
                    className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-500 ${
                      isRecording 
                        ? 'bg-primary text-black scale-110 shadow-[0_0_30px_rgba(6,182,212,0.5)]' 
                        : 'bg-white/10 text-white hover:bg-white/20 hover:scale-105'
                    }`}
                  >
                    {isProcessing ? (
                      <div className="animate-spin h-6 w-6 border-2 border-black/30 border-t-black rounded-full"></div>
                    ) : isRecording ? (
                      <Square className="w-6 h-6 fill-current" />
                    ) : (
                      <Mic className="w-6 h-6" />
                    )}
                  </button>
               </div>
            </div>

            <div className="max-w-md text-center space-y-4">
              <h2 className="text-2xl font-light text-white/90">
                {isRecording ? "I'm listening..." : isProcessing ? "Processing..." : "Tap to speak"}
              </h2>
              <p className="text-white/40 text-sm leading-relaxed">
                I adapt to your voice and emotions over time. Speak naturally.
              </p>
            </div>
          </div>

          {/* Right: Chat Log */}
          <div className="w-full lg:w-[450px] bg-surface/50 border-l border-white/5 flex flex-col h-full max-h-screen">
            <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
              {currentChat.messages.length === 0 && (
                <div className="h-full flex items-center justify-center text-white/20 text-sm uppercase tracking-widest">
                  No conversation yet
                </div>
              )}
              
              {currentChat.messages.map((msg, idx) => (
                <div key={idx} className={`flex flex-col gap-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  
                  <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed ${
                    msg.role === 'user' 
                      ? 'bg-white/10 text-white rounded-tr-sm' 
                      : 'bg-primary/10 text-primary-foreground rounded-tl-sm border border-primary/20'
                  }`}>
                    {msg.text}
                  </div>
                  
                  <div className="flex items-center gap-2 px-1">
                    {msg.role === 'user' ? (
                       <span className="text-[10px] uppercase tracking-wider text-white/30">{msg.emotion}</span>
                    ) : (
                       <div className="flex gap-2">
                          {msg.emotion && msg.emotion.slice(0, 2).map((emo, i) => (
                            <span key={i} className="text-[10px] uppercase tracking-wider text-primary/70">
                              {emo}
                            </span>
                          ))}
                       </div>
                    )}
                  </div>

                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
