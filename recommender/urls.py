from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('skip_movie/<int:movie_id>/', views.skip_movie, name='skip_movie'),
]