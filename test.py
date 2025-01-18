import requests
movie_id=862
url = "https://api.themoviedb.org/3/movie/{}?api_key=eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3ODExNzM4NzMwMzFlN2RmOGM2NDU1MjUwZWExYzBlOCIsIm5iZiI6MTczNzE2MzMzNC4yMjIwMDAxLCJzdWIiOiI2NzhiMDI0NjQyZjI3Yzc1NGM2NGQ5NDUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.rufF6MoVX-WLPkwcJTjIzFJ4nQD-spkPs_DTl_iFezQ&language=en-US".format(movie_id)

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3ODExNzM4NzMwMzFlN2RmOGM2NDU1MjUwZWExYzBlOCIsIm5iZiI6MTczNzE2MzMzNC4yMjIwMDAxLCJzdWIiOiI2NzhiMDI0NjQyZjI3Yzc1NGM2NGQ5NDUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.rufF6MoVX-WLPkwcJTjIzFJ4nQD-spkPs_DTl_iFezQ"
}

response = requests.get(url, headers=headers)
response=response.json()
print("https://image.tmdb.org/t/p/w185"+response['poster_path'])
# url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)