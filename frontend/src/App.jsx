import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchInterface from './components/SearchInterface';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/search" element={<SearchInterface />} />
        <Route path="/" element={<SearchInterface />} />
      </Routes>
    </Router>
  );
}

export default App;
