import base64
import pickle
import secrets
import tokenize
import uuid

from django.core.files.uploadedfile import UploadedFile
from django.shortcuts import render
from django.http import HttpResponse, HttpRequest

from generator.engine.midigenerator import generate_from_binary_str


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def upload_midi(request: HttpRequest):
    # Обработка файла
    if request.method == 'POST':
        uploaded_file = request.body
        token = str(uuid.uuid4())
        generate_from_binary_str(base64.b64encode(uploaded_file).decode('utf-8'), 10, token)
        return HttpResponse(token)
    else:
        return HttpResponse('Ошибка: только POST запросы разрешены')
