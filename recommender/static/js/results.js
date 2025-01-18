document.addEventListener('DOMContentLoaded', () => {
    let posters = document.querySelectorAll('.movie-poster')
    posters.forEach(poster => {
        poster.style.opacity = '0'
        poster.style.transition = 'opacity 0.5s ease'
        poster.onload = function() {
            poster.style.opacity = '1'
        }
    })
})
