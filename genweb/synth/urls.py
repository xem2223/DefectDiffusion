# synth/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),          # 홈
    path("project/", views.project, name="project"),  # ✅ 프로젝트 소개
    path("generate/", views.generator, name="generate"),
]
