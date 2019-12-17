from django.urls import path
from cvweb.views import index,upload

urlpatterns = [
    path(r'index/', index),
    path(r'upload/', upload,name="image")
]
