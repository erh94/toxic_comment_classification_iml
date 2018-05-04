from django.db import models
from django.utils import timezone


class ModelInput(models.Model):
    comment = models.TextField()

    def __str__(self):
        return self.comment
