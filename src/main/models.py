from django.db import models


from django.db import models
from django.utils import timezone


# Create your models here.
class Question(models.Model):
    question = models.CharField(max_length=1000)
    question_class = models.CharField(max_length=200)
    answer = models.IntegerField()
    date = models.DateTimeField('date asked', default=None)
