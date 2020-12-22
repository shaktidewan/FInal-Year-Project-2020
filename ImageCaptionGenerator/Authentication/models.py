from django.db import models


# Create your models here.
class Datas(models.Model):
    	caption = models.CharField(max_length=400)
    	img = models.ImageField(upload_to='pics')

    