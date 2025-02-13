import React, { useState } from 'react';
import { Container, Form, Button, Card, Alert } from 'react-bootstrap';
import { Search } from 'lucide-react';
import 'bootstrap/dist/css/bootstrap.min.css';

const SearchInterface = () => {
  const [query, setQuery] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setSummary('');

    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          n_results: 5
        }),
      });

      if (!response.ok) {
        throw new Error('Error en la búsqueda');
      }

      const data = await response.json();
      setSummary(data.resumen || 'No se encontró resumen');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ backgroundColor: '#1a1a1a', minHeight: '100vh' }} className="d-flex justify-content-center align-items-center">
      <Container className="d-flex flex-column justify-content-center align-items-center text-center w-100" style={{ maxWidth: '600px' }}>
        <h1 className="text-white mb-4">GPT Entrevistas</h1>
        
        <Form onSubmit={handleSearch} className="w-100">
          <div className="d-flex gap-2">
            <Form.Control
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="¿Qué quieres saber sobre los candidatos?"
              disabled={loading}
              className="bg-dark text-white border-secondary"
              style={{ flex: 1 }}
            />
            <Button 
              type="submit" 
              variant="dark"
              disabled={loading}
              className="d-flex align-items-center gap-2"
            >
              <Search size={20} />
              {loading ? 'Buscando...' : 'Buscar'}
            </Button>
          </div>
        </Form>

        {error && (
          <Alert variant="danger" className="mt-3">
            {error}
          </Alert>
        )}

        {summary && (
          <Card bg="dark" text="white" className="mt-3 w-100">
            <Card.Body>
              <div style={{ whiteSpace: 'pre-line' }}>{summary}</div>
            </Card.Body>
          </Card>
        )}
      </Container>
    </div>
  );
};

export default SearchInterface;
