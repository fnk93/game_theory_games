from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    #url(r'^index/$', views.index, name='index'),
    url(r'^game/$', views.game, name='game'),
    url(r'^solution/$', views.solution, name='solution'),
    url(r'^pdf_game/$', views.pdf_game, name='pdf_game'),
    url(r'^pdf_solution/$', views.pdf_solution, name='pdf_solution'),
    url(r'^latex_game/$', views.latex_game, name='latex_game'),
    url(r'^latex_solution/$', views.latex_solution, name='latex_solution'),
]
