// Global variables
let currentOffset = 0;
let currentType = 'similar';
let currentSelection = null;

// Show/hide loading spinner
function toggleLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'block' : 'none';
}

// Show error message
function showError(message) {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = `
        <div class="col-12 text-center">
            <div class="alert alert-danger" role="alert">
                ${message}
            </div>
        </div>
    `;
}

// Display recommendations
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = '';

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = `
            <div class="col-12 text-center">
                <div class="alert alert-info" role="alert">
                    No recommendations found. Try another selection!
                </div>
            </div>
        `;
        return;
    }

    recommendations.forEach((movie, index) => {
        const movieCard = document.createElement('div');
        movieCard.className = 'col-md-4 mb-4';
        const posterUrl = movie.info.basic_info.poster_path ?
            `https://image.tmdb.org/t/p/w500${movie.info.basic_info.poster_path}` :
            '/static/images/no-poster.jpg';

        movieCard.innerHTML = `
            <div class="movie-card animate__animated animate__fadeIn" style="animation-delay: ${index * 0.1}s">
                <img src="${posterUrl}" 
                     class="card-img-top" 
                     alt="${movie.title}"
                     onerror="this.src='/static/images/no-poster.jpg'">
                <div class="card-body">
                    <h5 class="card-title">${movie.title}</h5>
                    <p class="card-text">Rating: ${movie.info.basic_info.vote_average}/10</p>
                    <div class="d-flex justify-content-between">
                        <a href="/movie_details/${encodeURIComponent(movie.title)}" 
                           class="btn btn-primary btn-sm">View Details</a>
                        <button class="btn btn-outline-primary btn-sm add-to-watchlist" 
                                data-movie="${movie.title}">Add to Watchlist</button>
                    </div>
                </div>
            </div>
        `;
        container.appendChild(movieCard);
    });

    // Show refresh button
    document.querySelector('.refresh-container').style.display = 'block';
}

// Get recommendations
async function getRecommendations(type, selection) {
    try {
        toggleLoading(true);

        const response = await fetch('/get_recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: type,
                movie: type === 'similar' ? selection : null,
                mood: type === 'mood' ? selection : null,
                season: type === 'seasonal' ? selection : null,
                offset: currentOffset
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error getting recommendations');
        }

        currentType = type;
        currentSelection = selection;
        displayRecommendations(data.recommendations);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Error getting recommendations. Please try again.');
    } finally {
        toggleLoading(false);
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Movie Search
    document.getElementById('getRecommendations')?.addEventListener('click', () => {
        const movieTitle = document.getElementById('movieSelect').value;
        if (!movieTitle) {
            showError('Please select a movie first');
            return;
        }
        currentOffset = 0;
        getRecommendations('similar', movieTitle);
    });

    // Mood Based
    document.getElementById('getMoodRecommendations')?.addEventListener('click', () => {
        const mood = document.getElementById('moodSelect').value;
        if (!mood) {
            showError('Please select a mood first');
            return;
        }
        currentOffset = 0;
        getRecommendations('mood', mood);
    });

    // Seasonal
    document.getElementById('getSeasonalRecommendations')?.addEventListener('click', () => {
        const season = document.getElementById('seasonSelect').value;
        if (!season) {
            showError('Please select a season first');
            return;
        }
        currentOffset = 0;
        getRecommendations('seasonal', season);
    });

    // Refresh recommendations
    document.getElementById('refreshRecommendations')?.addEventListener('click', () => {
        currentOffset += 6;
        getRecommendations(currentType, currentSelection);
    });

    // Watchlist functionality
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('add-to-watchlist')) {
            addToWatchlist(e.target.dataset.movie);
        }
    });
});

// Add to watchlist
async function addToWatchlist(movieTitle) {
    try {
        const response = await fetch('/add_to_watchlist', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                movie: movieTitle
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Show success message
            const toast = document.createElement('div');
            toast.className = 'toast show position-fixed bottom-0 end-0 m-3';
            toast.setAttribute('role', 'alert');
            toast.innerHTML = `
                <div class="toast-header">
                    <strong class="me-auto">Success</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${data.message}
                </div>
            `;
            document.body.appendChild(toast);

            // Remove toast after 3 seconds
            setTimeout(() => toast.remove(), 3000);
        } else {
            throw new Error(data.error || 'Error adding to watchlist');
        }
    } catch (error) {
        console.error('Error:', error);
        alert(error.message || 'Error adding to watchlist');
    }
}