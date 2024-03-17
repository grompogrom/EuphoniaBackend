import base64
import uuid

from django.http import HttpResponse, HttpRequest, HttpResponseNotFound

from generator.engine.midigenerator import generate_from_binary_str

from euphonia.tools import get_file


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def upload_midi(request: HttpRequest):
    if request.method == 'POST':
        uploaded_file = request.body
        token = str(uuid.uuid4())
        generate_from_binary_str(base64.b64encode(uploaded_file).decode('utf-8'), 10, token)
        return HttpResponse(token)
    else:
        return HttpResponse('Ошибка: только POST запросы разрешены')


def get_generated(request: HttpRequest, token):
    if request.method != "GET":
        return HttpResponseNotFound()
    file = get_file(token)
    if file is None:
        return HttpResponse("File not exits or not ready", status=202)
    with open(file, 'rb') as f:
        file_data = f.read()
    response = HttpResponse(file_data, content_type='audio/midi')
    response['Content-Disposition'] = f'attachment; filename="{file}"'
    return response
