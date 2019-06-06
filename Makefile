default:	kmeans.cpp
	g++ kmeans.cpp -o kmeans
	
dbg:	kmeans.cpp
	g++ kmeans.cpp -o kmeans -D DBG
