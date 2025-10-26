# generator/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'), 
    
    path('generate/', views.generate_view, name='generate'), 
    
    # 실제 Project 페이지가 있다면 해당 뷰를 정의해야 합니다. 여기서는 임시로 home으로 연결합니다.
    path('project/', views.project_view, name='project'), 
]
