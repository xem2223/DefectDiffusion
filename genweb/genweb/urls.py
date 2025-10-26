# genweb/urls.py

from django.contrib import admin
from django.urls import path, include  # include 함수를 추가로 import 합니다.
from django.conf import settings        # media 파일 서빙을 위해 settings import
from django.conf.urls.static import static # media 파일 서빙을 위해 static 함수 import

urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('', include('generator.urls')), 
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROO
