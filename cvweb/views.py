from django.shortcuts import render

# Create your views here.
from rest_framework import status
import time
from rest_framework.decorators import api_view
from rest_framework.response import Response
from Cvdeeplabv3p.settings import MEDIA_ROOT, WEB_HOST_MEDIA_URL
from django.http.response import JsonResponse
from django.http import HttpResponse
from django.shortcuts import render, redirect
from PIL import Image
from io import BytesIO
import base64, json
from cvweb.Script.Deep import run_visualization
from django.views.decorators.csrf import csrf_exempt,csrf_protect,ensure_csrf_cookie

def index(request):
    return render(request, 'index.html')

@csrf_protect
def upload(request):
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        model = request.POST.get('select')
        data = request.FILES.get('explorer', "None")
        original_im = Image.open(data.file)
        dd = run_visualization(original_im)
        name = str(int(time.time()))
        dd.save(MEDIA_ROOT + '/{}.png'.format(name))
        data = {}
        data['urls'] = WEB_HOST_MEDIA_URL + '{}.png'.format(name)
        return JsonResponse(data)
