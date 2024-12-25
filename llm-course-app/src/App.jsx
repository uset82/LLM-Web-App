import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Home from './pages/Home';
import WeekContent from './pages/WeekContent';
import Labs from './pages/Labs';
import References from './pages/References';
import CharlyChatbot from './components/chatbot/CharlyChatbot';

function App() {
  return (
    <Router>
      <div className="flex h-screen bg-background">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Navbar />
          <main className="flex-1 overflow-auto">
            <div className="container py-6 space-y-6">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/week/:weekId/content" element={<WeekContent />} />
                <Route path="/week/:weekId/labs" element={<Labs />} />
                <Route path="/references" element={<References />} />
              </Routes>
            </div>
          </main>
        </div>
        <CharlyChatbot />
      </div>
    </Router>
  );
}

export default App;
