import re
from PyPDF2 import PdfReader

# PDF pour lequel on souhaite retrouver chaque référence
pdf_path = "./essai.pdf"
doc = PdfReader(pdf_path)

# On cherche toutes les références trouvées dans le texte passé en paramètre
def TrouveReference(text):
	xp_code_ref = r"~~(.*?)~~" # Cherche tout le texte encadré par ~~...~~, c’est-à-dire les références
	code = re.findall(xp_code_ref, text, re.DOTALL)
	return code

def Reference(doc, output_file="reference.txt"):
	# On récupère le nombre de pages 
	nombre_de_pages = len(doc.pages)
	# Compteur du nombre de référence pour l'ensemble du PDF
	compteur = 0
	with open(output_file, "w", encoding="utf-8") as ref :
		ref.write(f"Liste des références\n\n")
		for i in range(0, nombre_de_pages): 
			# On accède à chaque page du pdf
			page = doc.pages[i]
			# On extrait le texte de la page
			text = page.extract_text()
			# Affichage de la page traitée
			ref.write(f"Page {i+1} \n\n")
			# code contient toute les références de la page en cours de traitement
			code = TrouveReference(text)
			# On enlève tous les retours à la ligne qu'on peut trouver dans chaque référence
			code = [c.replace("\n","") for c in code]
			# Affichage des références trouvées dans la page en cours de traitement
			for j in range(len(code)):
				compteur += 1
				ref.write(f"Référence {compteur} : {code[j]}\n")
			ref.write("\n")
	return
if __name__ == "__main__":
	Reference(doc)


