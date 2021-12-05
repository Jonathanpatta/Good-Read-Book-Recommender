from django.http.response import JsonResponse
from django.shortcuts import render
from django.forms.models import model_to_dict
from django.contrib import messages

from django.core.cache import cache

from .recommend import *

from .models import Book

import json

# Create your views here.


def home(request):

    query = request.GET.get("query")
    books = Book.objects.order_by('-rating').filter(genre='Non-Fiction')[:100]


    cached_val = cache.get(query)
    if query:
        books = (Book.objects.filter(title__icontains=query) | Book.objects.filter(description__icontains=query))[:100]
        messages.info(request,f"searching for {query}")

        # books = books.filter(genre='Non-Fiction')

        print(messages.get_messages(request))
        
    


    

    

    return render(request,'main/home.html',{"books":books})


def book(request,pk):
    book = Book.objects.get(pk=pk)

    cached_val = cache.get(book.title)
    if cached_val:
        print("cache hit!!")
        recs = json.loads(cached_val)
    
    
    else:
        if book:
            recs = recommendations(book.title,df,model)
            cache.set(book.title,json.dumps(recs),500)


    books = Book.objects.none()
    if recs:
        for title in recs:
            books = books | Book.objects.filter(title=title)


    
    
    return render(request,'main/book.html',{"book":book,"recommendations":books})


def search(request):
    query = request.GET.get("q")
    
    if query:
        books = Book.objects.filter(title__icontains=query) | Book.objects.filter(description__icontains=query)

    

    book_list = []

    for book in books:
        book_list.append(model_to_dict(book))

   
    return JsonResponse(book_list,safe=False)


def recommend(request):
    query = request.GET.get("q")
    recs = []

    
    if query:
        recs = recommendations(query,df,model)

    books = Book.objects.none()
    if recs:
        for title in recs:
            books = books | Book.objects.filter(title=title)


    book_list = []

    for book in books:
        book_list.append(model_to_dict(book))

   
    return JsonResponse(book_list,safe=False)
