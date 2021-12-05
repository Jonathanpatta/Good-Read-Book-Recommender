from django.db import models
from django.db.models.fields.files import ImageField

# Create your models here.

class Book(models.Model):
    title = models.TextField(blank=True,null=True)
    author = models.CharField(blank=True,null=True,max_length=250)
    genre = models.CharField(blank=True,null=True,max_length=250)
    rating = models.FloatField(blank=True,null=True)
    description = models.TextField(blank=True,null=True)
    image_url = models.URLField(blank=True,null=True)
