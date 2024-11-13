// Global variables
let currentOffset = 0;
let currentType = 'similar';
let currentSelection = null;

// Helper function to show loading state
function showLoading() {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = '<div class="loading">Loading recommendations...</div>';
}

// Helper function to show error
function showError(message) {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = `<div class="error-message">${message}</div>`;
}

// Helper function to display recommendations
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = '';

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="text-center">No recommendations found</p>';
        return;
    }

    recommendations.forEach(movie => {
        const movieCard = document.createElement('div');
        movieCard.className = 'col-md-4 mb-4';
        const posterUrl = movie.info.basic_info.poster_path ?
            `https://image.tmdb.org/t/p/w500${movie.info.basic_info.poster_path}` :
            '/static/images/no-poster.jpg';

        movieCard.innerHTML = `
            <div class="card movie-card">
                <img src="${posterUrl}" 
                     class="card-img-top" 
                     alt="${movie.title}"
                     onerror="this.src='/static/images/no-poster.jpg'">
                <div class="card-body">
                    <h5 class="card-title">${movie.title}</h5>
                    <p class="card-text">Rating: ${movie.info.basic_info.vote_average}</p>
                    <div class="d-flex justify-content-between">
                        <button class="btn btn-sm btn-primary view-details" 
                                data-movie="${movie.title}">View Details</button>
                        <button class="btn btn-sm btn-outline-primary add-watchlist" 
                                data-movie="${movie.title}">Add to Watchlist</button>
                    </div>
                </div>
            </div>
        `;
        container.appendChild(movieCard);
    });

    // Show refresh button
    document.querySelector('.refresh-btn').style.display = 'block';
}

// Function to get recommendations
async function getRecommendations(type, selection) {
    try {
        showLoading();

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

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            showError(data.error);
            return;
        }

        displayRecommendations(data.recommendations);
        document.querySelector('.refresh-btn').style.display = 'block';

    } catch (error) {
        console.error('Error:', error);
        showError('Error getting recommendations. Please try again.');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Similar movies recommendations
    document.getElementById('getRecommendations').addEventListener('click', () => {
        const movieTitle = document.getElementById('movieSelect').value;
        if (!movieTitle) {
            showError('Please select a movie first');
            return;
        }
        currentType = 'similar';
        currentSelection = movieTitle;
        getRecommendations('similar', movieTitle);
    });

    // Mood based recommendations
    document.getElementById('getMoodRecommendations').addEventListener('click', () => {
        const mood = document.getElementById('moodSelect').value;
        if (!mood) {
            showError('Please select a mood first');
            return;
        }
        currentType = 'mood';
        currentSelection = mood;
        getRecommendations('mood', mood);
    });

    // Seasonal recommendations
    document.getElementById('getSeasonalRecommendations').addEventListener('click', () => {
        const season = document.getElementById('seasonSelect').value;
        if (!season) {
            showError('Please select a season first');
            return;
        }
        currentType = 'seasonal';
        currentSelection = season;
        getRecommendations('seasonal', season);
    });

    // Refresh recommendations
    document.getElementById('refreshRecommendations').addEventListener('click', () => {
        currentOffset += 6;  // Increment offset
        getRecommendations(currentType, currentSelection);
    });

    // View details buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('view-details')) {
            const movieTitle = e.target.dataset.movie;
            window.location.href = `/movie_details/${encodeURIComponent(movieTitle)}`;
        }
    });

    // Add to watchlist buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('add-watchlist')) {
            const movieTitle = e.target.dataset.movie;
            addToWatchlist(movieTitle);
        }
    });
});

// Function to add movie to watchlist
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
        alert(data.message);

    } catch (error) {
        console.error('Error:', error);
        alert('Error adding movie to watchlist');
    }
}