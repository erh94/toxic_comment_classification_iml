# Generated by Django 2.0.2 on 2018-05-02 13:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModelInput',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('originalSentence', models.CharField(max_length=200)),
                ('unl', models.TextField()),
            ],
        ),
    ]
