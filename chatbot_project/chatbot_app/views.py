from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import render
from .models import PDFDocument
from .serializers import PDFDocumentSerializer
from chatbot_app.langchain_setup import process_pdf, vector_store, generate_response
from rest_framework.parsers import MultiPartParser, FormParser


# Web Interface for Chatbot
def chatbot_page(request):
    return render(request, "chatbot.html")


# PDF Upload API
class PDFUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "file" not in request.FILES:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES["file"]
        if not file.name.endswith(".pdf"):
            return Response({"error": "Only PDF files allowed."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            pdf_document = PDFDocument.objects.create(title=file.name, file=file)
            
            # Process PDF Immediately
            process_pdf(pdf_document.file.path)  

            # Use Serializer to return structured data
            serializer = PDFDocumentSerializer(pdf_document)
            return Response(
                {"message": "PDF uploaded & processed.", "pdf": serializer.data},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Query Chatbot (Now Uses Llama-3)
class ChatbotQueryView(APIView):
    def get(self, request):
        query = request.GET.get("q", "").strip()
        if not query:
            return Response({"error": "No query provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve PDF embeddings from PGVector
        results = vector_store.similarity_search(query)
        context_text = "\n".join([r.page_content for r in results]) if results else "No matching documents."

        # Debugging: Print retrieved context
        print(f"Retrieved Context for Query: {query}\n{context_text}")

        # Generate AI response using Llama-3 (now returns a cleaned response)
        ai_response = generate_response(query)

        print(f"Final Extracted AI Response: {ai_response}")  # Debugging

        return Response({"results": ai_response}, status=status.HTTP_200_OK)


# Delete PDF & Embeddings
class PDFDeleteView(APIView):
    def delete(self, request, pdf_id):
        try:
            pdf = PDFDocument.objects.get(id=pdf_id)

            # Delete embeddings
            vector_store.delete([str(pdf_id)])

            # Delete the PDF
            pdf.file.delete(save=False)
            pdf.delete()

            return Response({"message": f"Deleted {pdf.title} & its embeddings."}, status=status.HTTP_200_OK)
        except PDFDocument.DoesNotExist:
            return Response({"error": "PDF not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# List PDFs
class PDFListView(APIView):
    def get(self, request):
        pdfs = PDFDocument.objects.all()
        serializer = PDFDocumentSerializer(pdfs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
