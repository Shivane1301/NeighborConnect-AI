import { useEffect, useState } from 'react'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8001'
const AUTH_STORAGE_KEY = 'neighborconnect_user'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail || 'The service is unavailable right now.')
  }
  return payload
}

function formatDistance(distance) {
  return `${Number(distance).toFixed(1)} km`
}

function VendorCard({ vendor }) {
  return (
    <article className="vendor-card">
      <div className="vendor-card__topline">
        <span className="category-label">{vendor.category}</span>
        <span className={`availability ${vendor.available ? 'is-available' : ''}`}>
          <span className="availability__dot" />
          {vendor.available ? 'Available now' : 'Currently busy'}
        </span>
      </div>
      <div className="vendor-card__heading">
        <div className="vendor-avatar">{vendor.name.charAt(0)}</div>
        <div>
          <h3>{vendor.name}</h3>
          <p className="vendor-location">{vendor.location}</p>
        </div>
      </div>
      <p className="vendor-description">{vendor.description}</p>
      <div className="vendor-stats">
        <span><strong>{vendor.rating.toFixed(1)}</strong> <span className="star">*</span> ({vendor.review_count})</span>
        <span>{formatDistance(vendor.distance)}</span>
        <span>{vendor.price}</span>
      </div>
      {vendor.recommendation_score !== undefined && (
        <div className="match-score">
          <span>Recommendation match</span>
          <strong>{Math.round(vendor.recommendation_score)}%</strong>
        </div>
      )}
      <div className="vendor-card__footer">
        <span>{vendor.phone}</span>
        <button type="button" className="text-button">View profile <span aria-hidden="true">-&gt;</span></button>
      </div>
    </article>
  )
}

function App() {
  const [authUser, setAuthUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY))
    } catch {
      return null
    }
  })
  const [authMode, setAuthMode] = useState(null)
  const [authForm, setAuthForm] = useState({ name: '', email: '', password: '', userType: 'customer' })
  const [authLoading, setAuthLoading] = useState(false)
  const [authError, setAuthError] = useState('')
  const [vendors, setVendors] = useState([])
  const [searchResults, setSearchResults] = useState(null)
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [vendorError, setVendorError] = useState('')
  const [searchError, setSearchError] = useState('')
  const [review, setReview] = useState('')
  const [sentiment, setSentiment] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [sentimentError, setSentimentError] = useState('')

  useEffect(() => {
    request('/api/vendors')
      .then(setVendors)
      .catch(() => setVendorError('We could not load local providers. Check that the backend is running.'))
  }, [])

  function openAuth(mode) {
    setAuthMode(mode)
    setAuthError('')
  }

  function closeAuth() {
    if (authLoading) return
    setAuthMode(null)
    setAuthError('')
  }

  function updateAuthField(field, value) {
    setAuthForm((current) => ({ ...current, [field]: value }))
  }

  async function handleAuth(event) {
    event.preventDefault()
    setAuthLoading(true)
    setAuthError('')
    try {
      const path = authMode === 'register' ? '/api/auth/register' : '/api/auth/login'
      const body = authMode === 'register'
        ? { name: authForm.name, email: authForm.email, password: authForm.password, role: authForm.userType }
        : { email: authForm.email, password: authForm.password }
      const user = await request(path, { method: 'POST', body: JSON.stringify(body) })
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(user))
      setAuthUser(user)
      setAuthMode(null)
      setAuthForm({ name: '', email: '', password: '', userType: 'customer' })
    } catch (error) {
      setAuthError(error.message)
    } finally {
      setAuthLoading(false)
    }
  }

  function handleLogout() {
    localStorage.removeItem(AUTH_STORAGE_KEY)
    setAuthUser(null)
  }

  async function handleSearch(event) {
    event.preventDefault()
    if (!query.trim()) return

    setSearching(true)
    setSearchError('')
    try {
      setSearchResults(await request('/api/search', {
        method: 'POST',
        body: JSON.stringify({ query: query.trim() }),
      }))
    } catch (error) {
      setSearchError(error.message)
      setSearchResults(null)
    } finally {
      setSearching(false)
    }
  }

  async function handleSentiment(event) {
    event.preventDefault()
    if (!review.trim()) return

    setAnalyzing(true)
    setSentimentError('')
    try {
      setSentiment(await request('/api/sentiment', {
        method: 'POST',
        body: JSON.stringify({ review: review.trim() }),
      }))
    } catch (error) {
      setSentimentError(error.message)
      setSentiment(null)
    } finally {
      setAnalyzing(false)
    }
  }

  const displayedVendors = searchResults || vendors

  return (
    <main className="app-shell">
      <nav className="topbar" aria-label="Main navigation">
        <a className="brand" href="#top" aria-label="NeighborConnect home">
          <span className="brand-mark">N</span>
          <span>Neighbor<span>Connect</span></span>
        </a>
        <div className="nav-links">
          <a href="#discover">Discover</a>
          <a href="#how-it-works">How it works</a>
          <a href="#reviews">Reviews</a>
        </div>
        {authUser ? (
          <div className="account-menu">
            <span className="account-name">Hi, {authUser.name}</span>
            <button type="button" className="nav-action" onClick={handleLogout}>Logout</button>
          </div>
        ) : (
          <button type="button" className="nav-action" onClick={() => openAuth('login')}>Sign in</button>
        )}
      </nav>

      <section className="hero" id="top">
        <div className="hero__copy">
          <p className="eyebrow">LOCAL HELP, HUMAN CONNECTION</p>
          <h1>Good help is closer than you think.</h1>
          <p className="hero__intro">Find trusted people in your neighborhood for the little things that keep life moving.</p>
          <form className="search-box" onSubmit={handleSearch}>
            <span className="search-icon" aria-hidden="true">/</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="What do you need help with?"
              aria-label="Search for a local service"
            />
            <button type="submit" disabled={searching || !query.trim()}>
              {searching ? 'Finding...' : 'Find help'}
            </button>
          </form>
          {searchError && <p className="error-message" role="alert">{searchError}</p>}
          <div className="popular-searches">
            <span>Try</span>
            {['Fix a leaking tap', 'House cleaning', 'Dinner for six'].map((item) => (
              <button type="button" key={item} onClick={() => setQuery(item)}>{item}</button>
            ))}
          </div>
        </div>
        <div className="hero__art" aria-hidden="true">
          <div className="sun-disc" />
          <div className="art-card art-card--one"><span>PLUMBING</span><strong>Tap fixed.<br />Day saved.</strong></div>
          <div className="art-card art-card--two"><span className="art-avatar">A</span><div><strong>Arun is nearby</strong><small>4.8 * &nbsp; 1.8 km away</small></div></div>
          <div className="hero-line hero-line--one" />
          <div className="hero-line hero-line--two" />
        </div>
      </section>

      <section className="section" id="discover">
        <div className="section-heading">
          <div>
            <p className="eyebrow">YOUR NEIGHBORHOOD NETWORK</p>
            <h2>{searchResults ? 'Best matches for you' : 'People ready to help'}</h2>
          </div>
          {searchResults && <button type="button" className="reset-button" onClick={() => setSearchResults(null)}>Show everyone</button>}
        </div>
        {vendorError && <p className="error-message" role="alert">{vendorError}</p>}
        {!vendorError && displayedVendors.length === 0 && <p className="empty-state">No providers found yet.</p>}
        <div className="vendor-grid">
          {displayedVendors.map((vendor) => <VendorCard key={vendor.id} vendor={vendor} />)}
        </div>
      </section>

      <section className="insight-section" id="reviews">
        <div className="insight-copy">
          <p className="eyebrow">A LITTLE EXTRA CLARITY</p>
          <h2>Words tell a story.</h2>
          <p>Paste a review and our language model will help you understand the feeling behind it.</p>
        </div>
        <form className="sentiment-panel" onSubmit={handleSentiment}>
          <label htmlFor="review-input">Analyze a review</label>
          <textarea
            id="review-input"
            value={review}
            onChange={(event) => setReview(event.target.value)}
            placeholder="The plumber was excellent and fixed everything quickly..."
            rows="4"
          />
          <div className="sentiment-panel__footer">
            <span>{review.length}/2000</span>
            <button type="submit" disabled={analyzing || !review.trim()}>{analyzing ? 'Analyzing...' : 'Read the feeling'}</button>
          </div>
          {sentimentError && <p className="error-message" role="alert">{sentimentError}</p>}
          {sentiment && (
            <div className="sentiment-result">
              <div className="sentiment-result__heading"><span className={`sentiment-pill sentiment-pill--${sentiment.sentiment.toLowerCase()}`}>{sentiment.sentiment}</span><strong>{Math.round(sentiment.confidence * 100)}% confidence</strong></div>
              <p>{sentiment.explanation}</p>
            </div>
          )}
        </form>
      </section>

      <footer className="footer" id="how-it-works">
        <a className="brand" href="#top"><span className="brand-mark">N</span><span>Neighbor<span>Connect</span></span></a>
        <p>Small tasks. Stronger neighborhoods.</p>
        <span className="footer-note">Made for your corner of the world</span>
      </footer>

      {authMode && (
        <div className="auth-backdrop" role="presentation" onMouseDown={closeAuth}>
          <section className="auth-modal" role="dialog" aria-modal="true" aria-labelledby="auth-title" onMouseDown={(event) => event.stopPropagation()}>
            <button type="button" className="auth-close" aria-label="Close authentication dialog" onClick={closeAuth}>x</button>
            <p className="eyebrow">WELCOME TO THE NEIGHBORHOOD</p>
            <h2 id="auth-title">{authMode === 'register' ? 'Create your account' : 'Welcome back'}</h2>
            <p className="auth-subtitle">{authMode === 'register' ? 'Join people making everyday life a little easier.' : 'Sign in to keep your neighborhood connections close.'}</p>
            <form className="auth-form" onSubmit={handleAuth}>
              {authMode === 'register' && (
                <label>Name<input required value={authForm.name} onChange={(event) => updateAuthField('name', event.target.value)} autoComplete="name" /></label>
              )}
              <label>Email<input required type="email" value={authForm.email} onChange={(event) => updateAuthField('email', event.target.value)} autoComplete="email" /></label>
              <label>Password<input required type="password" value={authForm.password} onChange={(event) => updateAuthField('password', event.target.value)} autoComplete={authMode === 'register' ? 'new-password' : 'current-password'} /></label>
              {authMode === 'register' && (
                <label>User type
                  <select value={authForm.userType} onChange={(event) => updateAuthField('userType', event.target.value)}>
                    <option value="customer">Customer</option>
                    <option value="vendor">Vendor</option>
                  </select>
                </label>
              )}
              {authError && <p className="auth-error" role="alert">{authError}</p>}
              <button type="submit" className="auth-submit" disabled={authLoading}>{authLoading ? 'Please wait...' : authMode === 'register' ? 'Create account' : 'Sign in'}</button>
            </form>
            <button type="button" className="auth-switch" onClick={() => { setAuthError(''); setAuthMode(authMode === 'register' ? 'login' : 'register') }}>
              {authMode === 'register' ? 'Already have an account? Sign in' : 'New here? Create account'}
            </button>
          </section>
        </div>
      )}
    </main>
  )
}

export default App
