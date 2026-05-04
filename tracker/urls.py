from django.urls import path
from . import views

urlpatterns = [
    path('',              views.home_view,           name='home'),
    path('register/',     views.register_view,        name='register'),
    path('login/',        views.login_view,           name='login'),
    path('logout/',       views.logout_view,          name='logout'),
    path('dashboard/',    views.dashboard_view,       name='dashboard'),
    # Single URL handles both CREATE and EDIT — view detects which automatically
    path('entry/',        views.mood_entry_view,      name='mood_entry'),
    path('result/<int:pk>/', views.result_view,       name='result'),
    path('history/',      views.history_view,         name='history'),
    path('weekly/',       views.weekly_summary_view,  name='weekly_summary'),
]
