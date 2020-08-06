#imports
import pytesseract
from pytesseract import Output
import pkg_resources
import cv2
from PIL import Image


#se configura el path donde se encuentra instalado tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR5.0\\tesseract.exe"

#imprime la version de tesseract
print(pkg_resources.working_set.by_key['pytesseract'].version)

#imprime la version de OpenCV
print(cv2.__version__)

#cargar la imagen que se quiere procesar
image_to_ocr = cv2.imread('images/libreta-individual.jpeg')
if image_to_ocr is None:
    exit();
#cv2.imshow("Imagen original",image_to_ocr)

#preprocesamiento de la imagen Paso 1: Convertir a escala de grises
preprocessed_img = cv2.cvtColor(image_to_ocr,cv2.COLOR_BGR2GRAY)

#preprocesamiento de la imagen Paso 2: Se aplica el filtro binario
preprocessed_img = cv2.adaptiveThreshold(preprocessed_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

#preprocesamiento de la imagen Paso 3: Se aplica un desenfocque a la imagen para remover ruido no deseado
#preprocessed_img = cv2.medianBlur(preprocessed_img, 3);

cv2.imshow("Imagen preprocesada",preprocessed_img)

#se guarda la imagen preprocesada en un archivo temporal
cv2.imwrite("test/temp_img.jpg",preprocessed_img)

#se carga la imagen con la libreria Pillow
preprocessed_pil_img = Image.open("images/libreta-individual.jpeg")

#procesar la iamgen co ocr
text_extracted = pytesseract.image_to_data(preprocessed_pil_img, output_type=Output.DICT);

print(text_extracted)

cv2.imshow("Imagen preprocesada",preprocessed_img)


