#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SpecificWordEmbeddings import SpecificWordEmbeddings

if __name__ == "__main__":
	
	def tokenize(x):
		res = []
		for i in range(len(x)): res.append(x[i].split(" "))
		return res
	
	x = ["La trama sigue a un robot llamado WALL·E, diseñado para limpiar la basura que cubre a la Tierra",
		 "después de que fuese devastada y abandonada por el ser humano en un futuro lejano. Tras esto, se enamora de EVE (EVA en Latinoamérica y España), ",
		 "una robot tipo sonda que es enviada al planeta para investigar si existen indicios de vida, lo cual significaría que el lugar puede ser nuevamente",
		 "habitado por la humanidad. Una vez que consigue su objetivo y encuentra una planta, EVA se dirige rápidamente a la nave de la que provino",
		 ", Axioma, por lo que WALL·E la sigue al espacio exterior en una aventura que cambia el destino de ambos para salvar a la naturaleza y a la humanidad."]

	y_train = [[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.]] # Categorical float
	x_train = tokenize(x)
	
	se  = SpecificWordEmbeddings(x_train, y_train, 16, 2, 128, 64, 2, 0.5, 10)
	se.train()
	se.save("./specific_model.bin", "./specific_model.tf")
	print(se.get_embedding("La"))
	se.load("./specific_model.bin", "./specific_model.tf")
	print(se.get_embedding("La"))
	
	#cbow = SpecificWordEmbeddings(x_train, None, 16, 2, 128, 64, 2, 0.5, 10)
	#cbow.train()
	#cbow.save("./cbow_model.bin", "./cbow_model.tf")
	#print(cbow.get_embedding("La"))
	#cbow.load("./cbow_model.bin", "./cbow_model.tf")
	#print(cbow.get_embedding("La"))
