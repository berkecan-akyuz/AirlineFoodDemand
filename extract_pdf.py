
import sys
import importlib.util

def check_install(package):
    spec = importlib.util.find_spec(package)
    return spec is not None

try:
    if check_install('pypdf'):
        from pypdf import PdfReader
    elif check_install('PyPDF2'):
        from PyPDF2 import PdfReader
    else:
        print("Neither pypdf nor PyPDF2 is installed.")
        # Try to install pypdf
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        from pypdf import PdfReader

    reader = PdfReader("SE390_Final_Project.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("requirements_extracted.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully extracted text to requirements_extracted.txt")

except Exception as e:
    print(f"Error: {e}")
