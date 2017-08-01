# try:
from NeuralNet import NeuralNet
# except Exception as e:
#
#     raise Exception("App Neural Net Not installed. This app requires Neural Net to be installed on the server.")

import os
import random

from django import http
from django.shortcuts import render
from django.utils import timezone

from . import firs
# Create your views here.
from .models import Question


# Create your views here.

def index(request):
    lastestQuestionList = Question.objects.order_by('-date')[:5]
    context = {
        'questions': lastestQuestionList,
    }
    return render(request, 'main/index.html', context)


def addToModel(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')
        # your tensorflow code here
        answer = firs.main(random.randint(0, 5), random.randint(0, 5))
        computed_result = NeuralNet.get_result(question)
        print('Question:', question)
        print('Answer  :', computed_result)
        print()
        q = Question(question=question, answer=answer, date=timezone.now())
        return http.JsonResponse(computed_result)

    return http.HttpResponse("not for direct call")
