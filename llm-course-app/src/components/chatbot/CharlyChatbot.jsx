import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageCircle, Minimize2, Maximize2, Send } from 'lucide-react';

const CharlyChatbot = () => {
  const [open, setOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!userInput.trim() || isLoading) return;

    // Add user message
    const newMessage = { role: 'user', content: userInput };
    setMessages(prev => [...prev, newMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      // Check if API key is missing or is the placeholder
      console.log('API Key status:', {
        exists: !!import.meta.env.VITE_GEMINI_API_KEY,
        value: import.meta.env.VITE_GEMINI_API_KEY
      });
      
      if (!import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.VITE_GEMINI_API_KEY === "YOUR_API_KEY") {
        throw new Error('INVALID_API_KEY');
      }

      // Initialize with basic course context while API endpoint is being fixed
      const courseContent = {
        content: `This is a course on Building LLM Applications covering:
        Week 1: Foundations of LLM Development
        Week 2: Advanced LLM Development
        Week 3: Production and Deployment
        Week 4: Advanced Applications
        
        The course includes hands-on labs, practical examples, and real-world applications.`
      };
      
      console.log('Sending request to Gemini API with content:', courseContent.content);
      
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${import.meta.env.VITE_GEMINI_API_KEY}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{
              role: "user",
              parts: [{ 
                text: `You are Charly, a helpful course assistant for the "Building LLM Applications" course. 
                Use the following course content to provide specific, relevant answers:
                
                ${courseContent.content || 'No specific course content found for this query.'}
                
                User Question: ${userInput}
                
                Remember to:
                1. Answer based on the course content provided
                2. If the question isn't related to the course content, politely redirect to course-relevant topics
                3. Include specific examples from the course when possible`
              }]
            }],
            generationConfig: {
              temperature: 0.7,
              topK: 40,
              topP: 0.95,
              maxOutputTokens: 1024,
            },
            safetySettings: [
              {
                category: "HARM_CATEGORY_HARASSMENT",
                threshold: "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                category: "HARM_CATEGORY_HATE_SPEECH",
                threshold: "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold: "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                category: "HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold: "BLOCK_MEDIUM_AND_ABOVE"
              }
            ]
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Gemini API error response:', errorData);
        throw new Error(`Failed to get response from Gemini API: ${errorData.error?.message || 'Unknown error'}`);
      }

      const data = await response.json();
      console.log('Gemini API response:', data);
      
      if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
        console.error('Unexpected response format:', data);
        throw new Error('Unexpected response format from Gemini API');
      }
      
      const content = data.candidates[0].content.parts[0].text;
      
      setMessages(prev => [...prev, { role: 'assistant', content }]);
    } catch (error) {
      console.error('Gemini API error:', error);
      const errorMessage = error.message === 'INVALID_API_KEY'
        ? 'I apologize, but I am currently unavailable as no valid API key is configured. Please contact the course administrator to set up the Gemini API key.'
        : 'I apologize, but I encountered an error while processing your request. Please try again.';
      
      setMessages(prev => [
        ...prev,
        { 
          role: 'assistant', 
          content: errorMessage
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Card className="fixed bottom-4 right-4 flex flex-col bg-background border shadow-lg z-50 w-80">
      <div className="flex items-center justify-between p-3 border-b">
        <div className="flex items-center space-x-2">
          <MessageCircle className="h-5 w-5" />
          <span className="font-medium">Charly</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setOpen(!open)}
          className="h-8 w-8 p-0"
        >
          {open ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
        </Button>
      </div>

      {open && (
        <div className="flex flex-col space-y-4 p-4">
          <ScrollArea className="h-[300px] pr-4" id="chat-messages">
            <div className="flex flex-col space-y-4">
              {messages.length === 0 ? (
                <div className="text-muted-foreground text-sm text-center py-4">
                  👋 Hi! I'm Charly, your course assistant. How can I help you today?
                </div>
              ) : (
                messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${
                      msg.role === 'assistant' ? 'justify-start' : 'justify-end'
                    }`}
                  >
                    <div
                      className={`rounded-lg px-4 py-2 max-w-[85%] ${
                        msg.role === 'assistant'
                          ? 'bg-muted text-foreground'
                          : 'bg-primary text-primary-foreground'
                      }`}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted text-foreground rounded-lg px-4 py-2">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          <div className="flex items-center space-x-2">
            <Input
              placeholder="Ask me anything..."
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              className="flex-1"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!userInput.trim() || isLoading}
              size="icon"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};

export default CharlyChatbot;
