# Generated by Django 3.2.6 on 2021-11-07 17:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='book',
            name='image',
        ),
        migrations.AddField(
            model_name='book',
            name='image_url',
            field=models.URLField(blank=True, null=True),
        ),
    ]
