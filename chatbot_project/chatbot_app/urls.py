from django.urls import path
from .views import PDFUploadView, ChatbotQueryView, PDFDeleteView, PDFListView, chatbot_page

urlpatterns = [
    path("", chatbot_page, name="chatbot_page"),
    path("api/upload/", PDFUploadView.as_view(), name="upload_pdf"),
    path("api/chat/", ChatbotQueryView.as_view(), name="chatbot_query"),
    path("api/delete/<int:pdf_id>/", PDFDeleteView.as_view(), name="delete_pdf"),
    path("api/list/", PDFListView.as_view(), name="list_pdfs"),
]
