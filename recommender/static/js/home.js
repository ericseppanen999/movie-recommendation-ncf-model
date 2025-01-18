function checkAllRatings() {
    let form = document.getElementById('ratings-form')
    let submitButton = document.getElementById('submit-button')
    let movieCards = document.querySelectorAll('.movie-card')
    let allRated = true

    movieCards.forEach(card => {
        let radioButtons = card.querySelectorAll('input[type="radio"]')
        let isRated = Array.from(radioButtons).some(radio => radio.checked)
        if (!isRated) {
            allRated = false
        }
    })

    if (allRated) {
        submitButton.classList.add('active')
    } else {
        submitButton.classList.remove('active')
    }
}

async function skipMovie(movieId) {
    let card = document.getElementById(`card-${movieId}`)
    let loadingOverlay = card.querySelector('.loading-overlay')
    let skipButton = card.querySelector('.skip-button')
    skipButton.disabled = true
    loadingOverlay.classList.add('active')
    try {
        let response = await fetch(`/skip_movie/${movieId}/`, {
            method: 'POST',
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            },
        })
        if (response.ok) {
            let newMovie = await response.json()
            if (newMovie.poster_url) {
                updateMovieCard(card, newMovie)
            } else {
                throw new Error('no poster')
            }
        } else {
            throw new Error('fetch error')
        }
    } catch (error) {
        console.error('error:', error)
        card.querySelector('.movie-title').innerHTML +=
            ' <span style="color: #ff4444;">(error loading)</span>'
    } finally {
        loadingOverlay.classList.remove('active')
        skipButton.disabled = false
    }
}

function updateMovieCard(card, newMovie) {
    let poster = card.querySelector('.movie-poster')
    let title = card.querySelector('.movie-title')
    let stars = card.querySelector('.stars')
    let skipButton = card.querySelector('.skip-button')

    let newImage = new Image()
    newImage.onload = function() {
        poster.src = newMovie.poster_url
    }
    newImage.src = newMovie.poster_url

    title.textContent = newMovie.title

    let radioButtons = stars.querySelectorAll('input[type="radio"]')
    radioButtons.forEach(radio => {
        radio.checked = false
        radio.name = `rating_${newMovie.movieId}`
        radio.id = `star_${newMovie.movieId}_${radio.value}`
    })

    let labels = stars.querySelectorAll('label')
    labels.forEach((label, index) => {
        label.htmlFor = `star_${newMovie.movieId}_${5 - index}`
    })

    skipButton.onclick = () => skipMovie(newMovie.movieId)

    stars.dataset.movieId = newMovie.movieId
    card.id = `card-${newMovie.movieId}`

    checkAllRatings()
}

function initializeSkipButtons() {
    document.querySelectorAll('.movie-card').forEach(card => {
        let movieId = card.id.split('-')[1]
        let skipButton = card.querySelector('.skip-button')
        skipButton.onclick = () => skipMovie(movieId)
    })
}

document.addEventListener('DOMContentLoaded', () => {
    checkAllRatings()
    initializeSkipButtons()
})
