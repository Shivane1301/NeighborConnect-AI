import { useEffect, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'
const AUTH_STORAGE_KEY = 'neighborconnect_user'
const SERVICE_CATEGORIES = ['Plumber', 'Electrician', 'Cleaner', 'Carpenter', 'Food Vendor', 'Appliance Repair']
const VENDOR_FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'available', label: 'Available Now' },
  { id: 'highly-rated', label: 'Highly Rated' },
  { id: 'nearby', label: 'Nearby' },
]

async function request(path, options = {}) {
  let savedUser = null
  try {
    savedUser = JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY))
  } catch {
    savedUser = null
  }
  const headers = { 'Content-Type': 'application/json' }
  if (savedUser?.access_token) headers.Authorization = `Bearer ${savedUser.access_token}`
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { ...headers, ...options.headers },
  })

  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail || 'The service is unavailable right now.')
  }
  return payload
}

function formatDistance(distance, isEstimate = false) {
  if (typeof distance !== 'number' || !Number.isFinite(distance)) return 'Distance unavailable'
  return `${isEstimate ? 'Approx. ' : ''}${distance.toFixed(1)} km away`
}

function formatSentiment(sentiment) {
  return sentiment ? `${sentiment.charAt(0).toUpperCase()}${sentiment.slice(1).toLowerCase()}` : 'Unknown'
}

function ReviewStars({ rating }) {
  return <span className="review-stars" aria-label={`${rating} out of 5 stars`}>{'★'.repeat(rating)}{'☆'.repeat(5 - rating)}</span>
}

function ReviewList({ reviews, emptyMessage }) {
  if (reviews.length === 0) return <p className="empty-state">{emptyMessage}</p>
  return (
    <div className="review-list">
      {reviews.map((item) => {
        const reviewText = item.review ?? item.text ?? ''
        const sentimentLabel = item.sentiment ? formatSentiment(item.sentiment) : 'Unknown'
        const confidence = typeof item.confidence === 'number' ? item.confidence.toFixed(2) : '0.00'
        return (
          <article className="review-row" key={item.id}>
            <div><ReviewStars rating={item.rating} /><span className={`sentiment-pill sentiment-pill--${item.sentiment}`}>{sentimentLabel}</span></div>
            <p>{reviewText}</p>
            <small>AI Sentiment: {sentimentLabel} | Confidence: {confidence}%</small>
            {item.created_at && <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleDateString()}</time>}
          </article>
        )
      })}
    </div>
  )
}

function VendorCard({ vendor, onViewProfile, onBook }) {
  const hasMatchScore = typeof vendor.recommendation_score === 'number' && Number.isFinite(vendor.recommendation_score)
  const matchedCategory = typeof vendor.matched_category === 'string' && vendor.matched_category.trim()
    ? vendor.matched_category
    : null
  const recommendationReasons = Array.isArray(vendor.recommendation_reasons)
    ? vendor.recommendation_reasons.filter((reason) => typeof reason === 'string' && reason.trim())
    : []
  const recommendationSummary = typeof vendor.recommendation_summary === 'string'
    ? vendor.recommendation_summary.trim()
    : ''

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
        <span><strong>{vendor.rating.toFixed(2)}</strong> <span className="star">*</span> ({vendor.review_count})</span>
        <span>{formatDistance(vendor.distance, vendor.distance_is_estimate)}</span>
        <span>{vendor.price}</span>
      </div>
      {(hasMatchScore || matchedCategory) && (
        <div className="vendor-card__recommendation-meta">
          {matchedCategory && <span className="matched-category">Matched: {matchedCategory}</span>}
          {hasMatchScore && (
            <div className="match-score">
              <span>Match Score</span>
              <strong>{vendor.recommendation_score.toFixed(2)} <small>Match</small></strong>
            </div>
          )}
        </div>
      )}
      {(recommendationReasons.length > 0 || recommendationSummary) && (
        <section className="recommendation-why" aria-label="Recommendation explanation">
          <h4>Why recommended</h4>
          {recommendationReasons.length > 0 && (
            <ul className="recommendation-reasons">
              {recommendationReasons.map((reason, index) => (
                <li className="recommendation-reason" key={`${reason}-${index}`}>
                  <span className="recommendation-reason__mark" aria-hidden="true">&#10003;</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          )}
          {recommendationSummary && <p className="recommendation-summary">{recommendationSummary}</p>}
        </section>
      )}
      <div className="vendor-card__footer">
        <span>{vendor.phone}</span>
        <div className="vendor-card__actions">
          <button type="button" className="text-button" onClick={() => onViewProfile(vendor)}>View profile <span aria-hidden="true">-&gt;</span></button>
          <button type="button" className="text-button" onClick={() => onBook(vendor)}>Request service</button>
        </div>
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
  const [authForm, setAuthForm] = useState({
    name: '', email: '', password: '', userType: 'customer', phone: '', location: '',
    latitude: '', longitude: '', category: 'Cleaner', description: '', serviceDetails: '',
    price: '', available: true, servicePreferences: [],
  })
  const [authLoading, setAuthLoading] = useState(false)
  const [authError, setAuthError] = useState('')
  const [vendors, setVendors] = useState([])
  const [searchResults, setSearchResults] = useState(null)
  const [vendorFilter, setVendorFilter] = useState('all')
  const [selectedVendor, setSelectedVendor] = useState(null)
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [vendorError, setVendorError] = useState('')
  const [searchError, setSearchError] = useState('')
  const [review, setReview] = useState('')
  const [sentiment, setSentiment] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [sentimentError, setSentimentError] = useState('')
  const [customerLocation, setCustomerLocation] = useState(() => authUser?.location || 'Puducherry')
  const [customerCoordinates, setCustomerCoordinates] = useState(null)
  const [locationMessage, setLocationMessage] = useState('Approximate distances use the selected area center.')
  const [maxPrice, setMaxPrice] = useState('')
  const [customerBookings, setCustomerBookings] = useState([])
  const [vendorDashboard, setVendorDashboard] = useState(null)
  const [vendorReviews, setVendorReviews] = useState([])
  const [bookingVendor, setBookingVendor] = useState(null)
  const [bookingDate, setBookingDate] = useState(() => new Date(Date.now() + 86400000).toISOString().slice(0, 16))
  const [bookingMessage, setBookingMessage] = useState('')
  const [bookingError, setBookingError] = useState('')
  const [actionError, setActionError] = useState('')
  const [reviewBooking, setReviewBooking] = useState(null)
  const [reviewText, setReviewText] = useState('')
  const [reviewRating, setReviewRating] = useState('5')
  const [reviewError, setReviewError] = useState('')
  const [reviewFeedback, setReviewFeedback] = useState(null)
  const [reviewSubmitting, setReviewSubmitting] = useState(false)
  const [vendorForm, setVendorForm] = useState(null)
  const [vendorSaveMessage, setVendorSaveMessage] = useState('')

  useEffect(() => {
    const params = new URLSearchParams()
    if (customerCoordinates) {
      params.set('latitude', customerCoordinates.latitude)
      params.set('longitude', customerCoordinates.longitude)
    } else if (customerLocation.trim()) {
      params.set('location', customerLocation.trim())
    }
    request(`/api/vendors${params.size ? `?${params}` : ''}`)
      .then(setVendors)
      .catch(() => setVendorError('We could not load local providers. Check that the backend is running.'))
  }, [customerLocation, customerCoordinates])

  useEffect(() => {
    if (!authUser) return
    if (authUser.role === 'customer') {
      request(`/api/bookings/${authUser.id}`).then(setCustomerBookings).catch(() => setCustomerBookings([]))
    } else if (authUser.role === 'vendor') {
      request('/api/vendor/dashboard').then((dashboard) => {
        setVendorDashboard(dashboard)
        setVendorForm(dashboard.vendor)
      }).catch(() => setVendorDashboard(null))
    }
  }, [authUser])

  useEffect(() => {
    const vendorId = selectedVendor?.id || vendorDashboard?.vendor?.id
    if (!vendorId) return
    request(`/api/reviews/${vendorId}`).then(setVendorReviews).catch(() => setVendorReviews([]))
  }, [selectedVendor, vendorDashboard?.vendor?.id])

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
        ? {
            name: authForm.name,
            email: authForm.email,
            password: authForm.password,
            role: authForm.userType,
            phone: authForm.phone,
            location: authForm.location,
            ...(authForm.latitude && authForm.longitude
              ? { latitude: Number(authForm.latitude), longitude: Number(authForm.longitude) }
              : {}),
            ...(authForm.userType === 'vendor'
              ? {
                  category: authForm.category,
                  description: authForm.description,
                  service_details: authForm.serviceDetails,
                  price: authForm.price,
                  available: authForm.available,
                }
              : { preferences: { services: authForm.servicePreferences } }),
          }
        : { email: authForm.email, password: authForm.password }
      const user = await request(path, { method: 'POST', body: JSON.stringify(body) })
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(user))
      setAuthUser(user)
      setCustomerLocation(user.location || 'Puducherry')
      setCustomerCoordinates(null)
      setAuthMode(null)
      setAuthForm({
        name: '', email: '', password: '', userType: 'customer', phone: '', location: '',
        latitude: '', longitude: '', category: 'Cleaner', description: '', serviceDetails: '',
        price: '', available: true, servicePreferences: [],
      })
    } catch (error) {
      setAuthError(error.message)
    } finally {
      setAuthLoading(false)
    }
  }

  function handleLogout() {
    localStorage.removeItem(AUTH_STORAGE_KEY)
    setAuthUser(null)
    setCustomerBookings([])
    setVendorDashboard(null)
    setReviewFeedback(null)
  }

  function useCurrentLocation() {
    setLocationMessage('Finding your location...')
    if (!navigator.geolocation) {
      setLocationMessage('Location is unavailable in this browser. Enter a city or area instead.')
      return
    }
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        setCustomerCoordinates({ latitude: coords.latitude, longitude: coords.longitude })
        setCustomerLocation('Current location')
        setLocationMessage('Using your current location for direct distance calculations.')
      },
      () => setLocationMessage('Location permission was denied. Enter a city or area to continue.'),
      { enableHighAccuracy: true, timeout: 10000 },
    )
  }

  function updateManualLocation(value) {
    setCustomerLocation(value)
    setCustomerCoordinates(null)
    setLocationMessage('Approximate distances use the selected area center.')
  }

  async function handleSearch(event, selectedFilter = vendorFilter) {
    event?.preventDefault()
    if (!query.trim()) return

    setVendorFilter('all')
    setSearching(true)
    setSearchError('')
    try {
      const body = {
        query: query.trim(),
        ...(customerCoordinates
          ? customerCoordinates
          : { location: customerLocation.trim() }),
        available_only: selectedFilter === 'available',
        nearby: selectedFilter === 'nearby',
        highly_rated: selectedFilter === 'highly-rated',
        ...(maxPrice ? { max_price: Number(maxPrice) } : {}),
      }
      setSearchResults(await request('/api/search', {
        method: 'POST',
        body: JSON.stringify(body),
      }))
    } catch (error) {
      setSearchError(error.message)
      setSearchResults([])
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

  async function handleCreateBooking(event) {
    event.preventDefault()
    setBookingError('')
    try {
      const booking = await request('/api/bookings', {
        method: 'POST',
        body: JSON.stringify({
          vendor_id: bookingVendor.id,
          service: bookingVendor.category,
          booking_date: new Date(bookingDate).toISOString(),
          message: bookingMessage,
        }),
      })
      setCustomerBookings((current) => [...current, booking])
      setBookingVendor(null)
      setBookingMessage('')
    } catch (error) {
      setBookingError(error.message)
    }
  }

  async function handleBookingStatus(bookingId, status) {
    setActionError('')
    try {
      const updated = await request(`/api/bookings/${bookingId}`, {
        method: 'PATCH', body: JSON.stringify({ status }),
      })
      if (authUser?.role === 'vendor') {
        setVendorDashboard((current) => ({
          ...current,
          requests: current.requests.map((booking) => booking.id === bookingId ? { ...booking, ...updated } : booking),
        }))
      } else {
        setCustomerBookings((current) => current.map((booking) => booking.id === bookingId ? { ...booking, ...updated } : booking))
      }
    } catch (error) {
      setActionError(error.message)
    }
  }

  async function handleReviewSubmit(event) {
    event.preventDefault()
    setReviewError('')
    setReviewSubmitting(true)
    try {
      const result = await request('/api/reviews', {
        method: 'POST',
        body: JSON.stringify({
          vendor_id: reviewBooking.vendor_id,
          booking_id: reviewBooking.id,
          rating: Number(reviewRating),
          review: reviewText,
        }),
      })
      const updatedVendor = result.updated_vendor
      const updatedReviews = await request(`/api/reviews/${reviewBooking.vendor_id}`)
      setVendors((current) => current.map((vendor) =>
        vendor.id === updatedVendor.id ? { ...vendor, ...updatedVendor } : vendor))
      setSearchResults((current) => current
        ? current.map((vendor) => vendor.id === updatedVendor.id ? { ...vendor, ...updatedVendor } : vendor)
        : current)
      setSelectedVendor((current) => current?.id === updatedVendor.id
        ? { ...current, ...updatedVendor }
        : current)
      setVendorReviews(updatedReviews)
      setReviewFeedback({
        ...result.review,
        sentiment: result.review?.sentiment,
        confidence: result.review?.confidence,
        updated_vendor: updatedVendor,
      })
      setCustomerBookings((current) => current.map((booking) =>
        booking.id === reviewBooking.id ? { ...booking, reviewed: true } : booking))
      setReviewBooking(null)
      setReviewText('')
    } catch (error) {
      setReviewError(error.message)
    } finally {
      setReviewSubmitting(false)
    }
  }

  async function saveVendorProfile(event) {
    event.preventDefault()
    setVendorSaveMessage('')
    try {
      const vendor = await request('/api/vendors/me', {
        method: 'PATCH',
        body: JSON.stringify({
          ...vendorForm,
          latitude: vendorForm.latitude ?? undefined,
          longitude: vendorForm.longitude ?? undefined,
        }),
      })
      setVendorDashboard((current) => ({ ...current, vendor }))
      setVendorForm(vendor)
      setVendorSaveMessage('Business profile updated.')
    } catch (error) {
      setVendorSaveMessage(error.message)
    }
  }

  async function toggleVendorAvailability() {
    try {
      const vendor = await request('/api/vendors/me', {
        method: 'PATCH',
        body: JSON.stringify({ available: !vendorDashboard.vendor.available }),
      })
      setVendorDashboard((current) => ({ ...current, vendor }))
      setVendorForm(vendor)
    } catch (error) {
      setVendorSaveMessage(error.message)
    }
  }

  function startBooking(vendor) {
    setBookingError('')
    setBookingDate(new Date(Date.now() + 86400000).toISOString().slice(0, 16))
    setBookingVendor(vendor)
    if (!authUser) setAuthMode('login')
  }

  const displayedVendors = searchResults || vendors
  const filteredVendors = displayedVendors.filter((vendor) => {
    if (vendorFilter === 'available') return vendor.available === true
    if (vendorFilter === 'highly-rated') return vendor.rating >= 4.5
    if (vendorFilter === 'nearby') return typeof vendor.distance === 'number' && vendor.distance <= 5
    if (maxPrice) {
      const prices = (vendor.price || '').match(/\d+(?:\.\d+)?/g)?.map(Number) || []
      if (!prices.length || Math.max(...prices) > Number(maxPrice)) return false
    }
    return true
  })
  const emptyVendorMessage = searchResults !== null
    ? `No${vendorFilter === 'available' ? ' available' : ''} matching service providers found. Try another area or preference.`
    : 'No providers found yet.'

  if (authUser?.role === 'vendor') {
    return (
      <main className="app-shell dashboard-shell">
        <nav className="topbar" aria-label="Main navigation">
          <a className="brand" href="#vendor-home"><span className="brand-mark">N</span><span>Neighbor<span>Connect</span></span></a>
          <span className="dashboard-nav-label">Business dashboard</span>
          <div className="account-menu"><span className="account-name">{vendorDashboard?.vendor?.name || authUser.name}</span><button type="button" className="nav-action" onClick={handleLogout}>Logout</button></div>
        </nav>
        <section className="dashboard-heading" id="vendor-home">
          <div><p className="eyebrow">YOUR LOCAL BUSINESS</p><h1>Welcome, {vendorDashboard?.vendor?.name || authUser.name}.</h1><p>Manage your profile, requests, availability, and customer reviews.</p></div>
          {vendorDashboard?.vendor && <button type="button" className={`availability-toggle${vendorDashboard.vendor.available ? ' is-on' : ''}`} onClick={toggleVendorAvailability}><span />{vendorDashboard.vendor.available ? 'Available now' : 'Currently unavailable'}</button>}
        </section>
        <section className="dashboard-columns">
          <section className="dashboard-section">
            <div className="section-heading"><div><p className="eyebrow">BUSINESS DETAILS</p><h2>My profile</h2></div></div>
            {vendorForm ? (
              <form className="profile-form" onSubmit={saveVendorProfile}>
                <label>Business name<input required value={vendorForm.name} onChange={(event) => setVendorForm((current) => ({ ...current, name: event.target.value }))} /></label>
                <label>Service category<select value={vendorForm.category} onChange={(event) => setVendorForm((current) => ({ ...current, category: event.target.value }))}>{SERVICE_CATEGORIES.map((category) => <option key={category}>{category}</option>)}</select></label>
                <label>Phone<input required value={vendorForm.phone} onChange={(event) => setVendorForm((current) => ({ ...current, phone: event.target.value }))} /></label>
                <label>Price range<input required value={vendorForm.price} onChange={(event) => setVendorForm((current) => ({ ...current, price: event.target.value }))} /></label>
                <label className="profile-form__wide">Description<textarea rows="3" value={vendorForm.description} onChange={(event) => setVendorForm((current) => ({ ...current, description: event.target.value }))} /></label>
                <label className="profile-form__wide">Service details<textarea rows="3" value={vendorForm.service_details || ''} onChange={(event) => setVendorForm((current) => ({ ...current, service_details: event.target.value }))} /></label>
                <label>Location<input required value={vendorForm.location} onChange={(event) => setVendorForm((current) => ({ ...current, location: event.target.value, latitude: null, longitude: null }))} /></label>
                <div className="profile-coordinate"><label>Latitude<input type="number" step="any" value={vendorForm.latitude ?? ''} onChange={(event) => setVendorForm((current) => ({ ...current, latitude: event.target.value === '' ? null : Number(event.target.value) }))} /></label><label>Longitude<input type="number" step="any" value={vendorForm.longitude ?? ''} onChange={(event) => setVendorForm((current) => ({ ...current, longitude: event.target.value === '' ? null : Number(event.target.value) }))} /></label></div>
                {vendorSaveMessage && <p className="form-feedback" role="status">{vendorSaveMessage}</p>}
                <button type="submit" className="auth-submit">Save business profile</button>
              </form>
            ) : <p className="empty-state">Loading your business profile...</p>}
            {vendorDashboard?.vendor && <div className="vendor-dashboard-stats"><span><strong>{vendorDashboard.vendor.rating.toFixed(2)}</strong> rating</span><span><strong>{vendorDashboard.vendor.review_count}</strong> reviews</span><span><strong>{vendorDashboard.vendor.price}</strong> price range</span></div>}
          </section>
          <section className="dashboard-section">
            <div className="section-heading"><div><p className="eyebrow">CUSTOMER BOOKINGS</p><h2>Service requests</h2></div><span className="dashboard-count">{vendorDashboard?.requests?.length || 0}</span></div>
            {actionError && <p className="error-message" role="alert">{actionError}</p>}
            {!vendorDashboard && <p className="empty-state">Loading requests...</p>}
            {vendorDashboard && vendorDashboard.requests.length === 0 && <p className="empty-state">New customer requests will appear here.</p>}
            <div className="request-list">{vendorDashboard?.requests?.map((booking) => (
              <article className="request-row" key={booking.id}>
                <div className="request-row__heading"><strong>{booking.service}</strong><span className={`request-status request-status--${booking.status}`}>{booking.status}</span></div>
                {booking.customer_name && <p>From {booking.customer_name}</p>}
                <p>{new Date(booking.booking_date).toLocaleString()}</p>
                {booking.message && <p>{booking.message}</p>}
                {booking.status === 'pending' && <button type="button" className="text-button" onClick={() => handleBookingStatus(booking.id, 'accepted')}>Accept request</button>}
                {booking.status === 'accepted' && <button type="button" className="text-button" onClick={() => handleBookingStatus(booking.id, 'completed')}>Mark completed</button>}
              </article>
            ))}</div>
          </section>
        </section>
        <section className="dashboard-section dashboard-reviews">
          <div className="section-heading"><div><p className="eyebrow">CUSTOMER FEEDBACK</p><h2>Reviews</h2></div></div>
            <ReviewList reviews={vendorReviews} emptyMessage="Reviews for completed services will appear here." />
        </section>
      </main>
    )
  }

  return (
    <main className="app-shell">
      <nav className="topbar" aria-label="Main navigation">
        <a className="brand" href="#top" aria-label="NeighborConnect home">
          <span className="brand-mark">N</span>
          <span>Neighbor<span>Connect</span></span>
        </a>
        <div className="nav-links">
          <a href="#discover">Discover</a>
          {authUser?.role === 'customer' && <a href="#my-requests">My requests</a>}
          <a href="#reviews">Reviews</a>
        </div>
        {authUser ? (
          <div className="account-menu">
            <span className="account-name">Hi, {authUser.name}</span>
            <button type="button" className="nav-action" onClick={handleLogout}>Logout</button>
          </div>
        ) : (
          <div className="guest-actions"><button type="button" className="text-button" onClick={() => openAuth('login')}>Sign in</button><button type="button" className="nav-action" onClick={() => openAuth('register')}>Sign up</button></div>
        )}
      </nav>

      <section className="hero" id="top">
        <div className="hero__copy">
          <p className="eyebrow">LOCAL HELP, HUMAN CONNECTION</p>
          <h1>{authUser?.role === 'customer' ? `Hi ${authUser.name.split(' ')[0]}, what service do you need?` : 'Good help is closer than you think.'}</h1>
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
          <div className="location-controls">
            <label htmlFor="customer-location">Your location</label>
            <input id="customer-location" value={customerLocation} onChange={(event) => updateManualLocation(event.target.value)} placeholder="Enter city or area, e.g. White Town" />
            <button type="button" className="text-button" onClick={useCurrentLocation}>Use my location</button>
            <p role="status">{locationMessage}</p>
          </div>
          <div className="service-shortcuts" aria-label="Browse service categories">
            {SERVICE_CATEGORIES.map((category) => {
              const examples = {
                Plumber: 'My kitchen tap is leaking', Electrician: 'Fix my ceiling fan', Cleaner: 'I need someone to clean my house',
                Carpenter: 'I need someone to repair my cupboard', 'Food Vendor': 'I need lunch for 10 people', 'Appliance Repair': 'My washing machine is not working',
              }
              return <button type="button" key={category} onClick={() => setQuery(examples[category])}>{category}</button>
            })}
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
            <h2>{searchResults !== null ? 'Best matches for you' : 'People ready to help'}</h2>
          </div>
          {searchResults && <button type="button" className="reset-button" onClick={() => { setSearchResults(null); setVendorFilter('all') }}>Show everyone</button>}
        </div>
        <div className="vendor-results-toolbar">
          <div className="vendor-filters" role="group" aria-label="Filter vendors">
            {VENDOR_FILTERS.map((filter) => (
              <button
                type="button"
                key={filter.id}
                className={`vendor-filter${vendorFilter === filter.id ? ' is-active' : ''}`}
                aria-pressed={vendorFilter === filter.id}
                onClick={() => {
                  setVendorFilter(filter.id)
                  if (query.trim() && searchResults !== null) handleSearch(null, filter.id)
                }}
              >
                {filter.label}
              </button>
            ))}
          </div>
          <p className="vendor-result-count" aria-live="polite">
            {filteredVendors.length} {filteredVendors.length === 1 ? 'vendor' : 'vendors'} found
          </p>
          <label className="price-filter">Price up to<select value={maxPrice} onChange={(event) => setMaxPrice(event.target.value)}><option value="">Any</option><option value="1000">Rs. 1,000</option><option value="1500">Rs. 1,500</option><option value="2500">Rs. 2,500</option></select></label>
        </div>
        {vendorError && <p className="error-message" role="alert">{vendorError}</p>}
        {!vendorError && filteredVendors.length === 0 && <p className="empty-state">{emptyVendorMessage}</p>}
        <div className="vendor-grid">
          {filteredVendors.map((vendor) => <VendorCard key={vendor.id} vendor={vendor} onViewProfile={setSelectedVendor} onBook={startBooking} />)}
        </div>
      </section>

      {authUser?.role === 'customer' && (
        <section className="section request-history" id="my-requests">
          <div className="section-heading"><div><p className="eyebrow">YOUR SERVICE ACTIVITY</p><h2>My requests</h2></div><span className="dashboard-count">{customerBookings.length}</span></div>
          {actionError && <p className="error-message" role="alert">{actionError}</p>}
          {reviewFeedback && <div className="review-feedback" role="status"><strong>Service review submitted to {reviewFeedback.updated_vendor.name}</strong><span>Vendor rating: {reviewFeedback.updated_vendor.rating.toFixed(2)} ({reviewFeedback.updated_vendor.review_count} reviews)</span><span>AI Sentiment: {formatSentiment(reviewFeedback.sentiment)} | Confidence: {reviewFeedback.confidence.toFixed(2)}%</span></div>}
          {customerBookings.length === 0 && <p className="empty-state">Your service requests will appear here after you book a provider.</p>}
          <div className="request-list request-list--customer">{customerBookings.map((booking) => (
            <article className="request-row" key={booking.id}>
              <div className="request-row__heading"><strong>{booking.vendor_name || `Provider #${booking.vendor_id}`}</strong><span className={`request-status request-status--${booking.status}`}>{booking.status}</span></div>
              <p>{booking.service} | {new Date(booking.booking_date).toLocaleString()}</p>
              {booking.message && <p>{booking.message}</p>}
              {booking.status === 'pending' && <button type="button" className="text-button" onClick={() => handleBookingStatus(booking.id, 'cancelled')}>Cancel request</button>}
              {booking.status === 'accepted' && !booking.reviewed && <p className="review-availability">Review available after service completion.</p>}
              {booking.status === 'completed' && !booking.reviewed && <button type="button" className="service-review-action" onClick={() => { setReviewBooking(booking); setReviewError('') }}>Write a review</button>}
              {booking.reviewed && <span className="reviewed-label">Review submitted</span>}
            </article>
          ))}</div>
        </section>
      )}

      <section className="insight-section" id="reviews">
        <div className="insight-copy">
          <p className="eyebrow">STANDALONE SENTIMENT DEMO</p>
          <h2>Analyze review sentiment.</h2>
          <p>Paste a review and our TF-IDF + Logistic Regression Machine Learning model will help identify its sentiment.</p>
        </div>
        <form className="sentiment-panel" onSubmit={handleSentiment}>
          <label htmlFor="review-input">Analyze a review</label>
          <textarea
            id="review-input"
            value={review}
            onChange={(event) => setReview(event.target.value)}
            placeholder="The plumber was excellent and fixed everything quickly..."
            rows="4"
            maxLength={2000}
          />
          <div className="sentiment-panel__footer">
            <span>{review.length}/2000</span>
            <button type="submit" disabled={analyzing || !review.trim()}>{analyzing ? 'Analyzing...' : 'Read the feeling'}</button>
          </div>
          {sentimentError && <p className="error-message" role="alert">{sentimentError}</p>}
          {sentiment && (
            <div className="sentiment-result">
              <div className="sentiment-result__heading"><span className={`sentiment-pill sentiment-pill--${sentiment.sentiment.toLowerCase()}`}>{sentiment.sentiment}</span><strong>{typeof sentiment.confidence === 'number' && Number.isFinite(sentiment.confidence) ? `Confidence: ${sentiment.confidence.toFixed(2)}%` : 'Confidence: Not available'}</strong></div>
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

      {bookingVendor && authUser?.role === 'customer' && (
        <div className="auth-backdrop" role="presentation" onMouseDown={() => setBookingVendor(null)}>
          <section className="auth-modal" role="dialog" aria-modal="true" aria-labelledby="booking-title" onMouseDown={(event) => event.stopPropagation()}>
            <button type="button" className="auth-close" aria-label="Close service request" onClick={() => setBookingVendor(null)}>x</button>
            <p className="eyebrow">NEW SERVICE REQUEST</p>
            <h2 id="booking-title">Request {bookingVendor.category.toLowerCase()}</h2>
            <div className="booking-summary"><strong>{bookingVendor.name}</strong><span>{bookingVendor.location}</span><span>{bookingVendor.price}</span><span>{bookingVendor.available ? 'Available now' : 'Currently busy'}</span></div>
            <form className="auth-form" onSubmit={handleCreateBooking}>
              <label>Preferred date and time<input required type="datetime-local" min={new Date().toISOString().slice(0, 16)} value={bookingDate} onChange={(event) => setBookingDate(event.target.value)} /></label>
              <label>Message (optional)<textarea rows="3" maxLength={2000} value={bookingMessage} onChange={(event) => setBookingMessage(event.target.value)} placeholder="Add details about the service" /></label>
              {bookingError && <p className="auth-error" role="alert">{bookingError}</p>}
              <button type="submit" className="auth-submit">Send service request</button>
            </form>
          </section>
        </div>
      )}

      {reviewBooking && (
        <div className="auth-backdrop" role="presentation" onMouseDown={() => setReviewBooking(null)}>
          <section className="auth-modal" role="dialog" aria-modal="true" aria-labelledby="review-title" onMouseDown={(event) => event.stopPropagation()}>
            <button type="button" className="auth-close" aria-label="Close review form" onClick={() => setReviewBooking(null)}>x</button>
            <p className="eyebrow">VENDOR SERVICE REVIEW</p>
            <h2 id="review-title">Review completed service</h2>
            <div className="booking-summary"><strong>{reviewBooking.vendor_name}</strong><span>{reviewBooking.service}</span><span>Completed {new Date(reviewBooking.booking_date).toLocaleString()}</span></div>
            <form className="auth-form" onSubmit={handleReviewSubmit}>
              <label>Star rating<select value={reviewRating} onChange={(event) => setReviewRating(event.target.value)}>{[5, 4, 3, 2, 1].map((rating) => <option key={rating} value={rating}>{'★'.repeat(rating)}{'☆'.repeat(5 - rating)} ({rating} stars)</option>)}</select></label>
              <label>Review text<textarea required rows="4" maxLength={2000} value={reviewText} onChange={(event) => setReviewText(event.target.value)} placeholder="Describe your experience with this service" /></label>
              {reviewError && <p className="auth-error" role="alert">{reviewError}</p>}
              <button type="submit" className="auth-submit" disabled={reviewSubmitting}>{reviewSubmitting ? 'Submitting...' : 'Submit Review'}</button>
            </form>
          </section>
        </div>
      )}

      {selectedVendor && (
        <div className="auth-backdrop" role="presentation" onMouseDown={() => setSelectedVendor(null)}>
          <section className="auth-modal vendor-profile-modal" role="dialog" aria-modal="true" aria-labelledby="vendor-profile-title" onMouseDown={(event) => event.stopPropagation()}>
            <button type="button" className="auth-close" aria-label="Close vendor details" onClick={() => setSelectedVendor(null)}>x</button>
            <p className="eyebrow">VENDOR DETAILS</p>
            <h2 id="vendor-profile-title">{selectedVendor.name}</h2>
            <dl className="vendor-details">
              <div><dt>Category</dt><dd>{selectedVendor.category}</dd></div>
              <div className="vendor-details__description"><dt>Description</dt><dd>{selectedVendor.description}</dd></div>
              <div><dt>Location</dt><dd>{selectedVendor.location}</dd></div>
              <div><dt>Distance</dt><dd>{formatDistance(selectedVendor.distance, selectedVendor.distance_is_estimate)}</dd></div>
              <div><dt>Rating</dt><dd>{selectedVendor.rating.toFixed(2)}</dd></div>
              <div><dt>Review count</dt><dd>{selectedVendor.review_count}</dd></div>
              <div><dt>Price</dt><dd>{selectedVendor.price}</dd></div>
              <div><dt>Availability</dt><dd>{selectedVendor.available ? 'Available now' : 'Currently busy'}</dd></div>
              <div><dt>Phone</dt><dd>{selectedVendor.phone}</dd></div>
              <div className="vendor-details__description"><dt>Service details</dt><dd>{selectedVendor.service_details || selectedVendor.description}</dd></div>
              {selectedVendor.recommendation_score !== undefined && (
                <div><dt>Recommendation score</dt><dd>{Math.round(selectedVendor.recommendation_score)}%</dd></div>
              )}
            </dl>
            <div className="profile-reviews"><h3>Customer reviews</h3><ReviewList reviews={vendorReviews} emptyMessage="No reviews yet." /></div>
            {authUser?.role !== 'vendor' && <button type="button" className="auth-submit profile-book-button" onClick={() => { setSelectedVendor(null); startBooking(selectedVendor) }}>Request service</button>}
          </section>
        </div>
      )}

      {authMode && (
        <div className="auth-backdrop" role="presentation" onMouseDown={closeAuth}>
          <section className="auth-modal" role="dialog" aria-modal="true" aria-labelledby="auth-title" onMouseDown={(event) => event.stopPropagation()}>
            <button type="button" className="auth-close" aria-label="Close authentication dialog" onClick={closeAuth}>x</button>
            <p className="eyebrow">WELCOME TO THE NEIGHBORHOOD</p>
            <h2 id="auth-title">{authMode === 'register' ? 'Create your account' : 'Welcome back'}</h2>
            <p className="auth-subtitle">{authMode === 'register' ? 'Join people making everyday life a little easier.' : 'Sign in to keep your neighborhood connections close.'}</p>
            <form className="auth-form" onSubmit={handleAuth}>
              {authMode === 'register' && <>
                <label>{authForm.userType === 'vendor' ? 'Business name' : 'Full name'}<input required value={authForm.name} onChange={(event) => updateAuthField('name', event.target.value)} autoComplete="name" /></label>
              </>}
              <label>Email<input required type="email" value={authForm.email} onChange={(event) => updateAuthField('email', event.target.value)} autoComplete="email" /></label>
              <label>Password<input required minLength={authMode === 'register' ? 8 : undefined} type="password" value={authForm.password} onChange={(event) => updateAuthField('password', event.target.value)} autoComplete={authMode === 'register' ? 'new-password' : 'current-password'} /></label>
              {authMode === 'register' && (
                <>
                  <label>Account type<select value={authForm.userType} onChange={(event) => updateAuthField('userType', event.target.value)}><option value="customer">Customer</option><option value="vendor">Vendor</option></select></label>
                  <label>Phone<input required value={authForm.phone} onChange={(event) => updateAuthField('phone', event.target.value)} autoComplete="tel" /></label>
                  <label>City or area<input required value={authForm.location} onChange={(event) => updateAuthField('location', event.target.value)} placeholder="e.g. White Town, Puducherry" /></label>
                  {authForm.userType === 'customer' ? (
                    <label>Service preferences (optional)<select multiple value={authForm.servicePreferences} onChange={(event) => updateAuthField('servicePreferences', Array.from(event.target.selectedOptions, (option) => option.value))}>{SERVICE_CATEGORIES.map((category) => <option key={category} value={category}>{category}</option>)}</select></label>
                  ) : <>
                    <label>Service category<select required value={authForm.category} onChange={(event) => updateAuthField('category', event.target.value)}>{SERVICE_CATEGORIES.map((category) => <option key={category}>{category}</option>)}</select></label>
                    <label>Description<textarea required rows="3" value={authForm.description} onChange={(event) => updateAuthField('description', event.target.value)} /></label>
                    <label>Service details<textarea rows="2" value={authForm.serviceDetails} onChange={(event) => updateAuthField('serviceDetails', event.target.value)} /></label>
                    <label>Price range<input required value={authForm.price} onChange={(event) => updateAuthField('price', event.target.value)} placeholder="Rs. 600-1500" /></label>
                    <label className="availability-checkbox"><input type="checkbox" checked={authForm.available} onChange={(event) => updateAuthField('available', event.target.checked)} />Available now</label>
                    <div className="profile-coordinate"><label>Latitude (optional)<input type="number" step="any" value={authForm.latitude} onChange={(event) => updateAuthField('latitude', event.target.value)} /></label><label>Longitude (optional)<input type="number" step="any" value={authForm.longitude} onChange={(event) => updateAuthField('longitude', event.target.value)} /></label></div>
                  </>}
                </>
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
