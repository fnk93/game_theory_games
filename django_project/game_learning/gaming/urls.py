from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^game/$', views.game, name='game'),
    url(r'^solution/$', views.solution, name='solution'),
    url(r'^latex_game/$', views.latex_game, name='latex_game'),
    url(r'^latex_solution/$', views.latex_solution, name='latex_solution'),
]
