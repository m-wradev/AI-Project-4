/* Author:		Ryan Armstrong
 * Project:		OLA 4 
 * Due:			December 4, 2018
 * Instructor: 	Dr. Phillips
 * Course:		CSCI 4350-001
 */

#include <iostream>
#include <utility>
#include <map>
#include <algorithm>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
#include <set>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

struct Dataset
{
	static int nFeatures;	// number of real-valued features in each dataset
	static int nClasses;	// number of unique classifications
	static set<int>* classifications;	// unique classifications

	vector<double> features;
	int classification;

	Dataset(string toParse)
	{
		classification = stoi(toParse.substr(toParse.rfind(" ")));
		string featstr = toParse.substr(0, toParse.rfind(" ") + 1);
		string feat = "";
		for (int i = 0; featstr[i] != '\0'; i++)
		{
			if (featstr[i] != ' ')
			{
				feat += featstr[i];
			}
			else
			{
				features.push_back(stod(feat));
				feat = "";
			}
		}
	}

	// two sets of data are the same if they have the same exact features
	bool operator==(Dataset* ds)
	{
		for (int i = 0; i < Dataset::nFeatures; i++)
			if (features[i] != ds->features[i])
				return false;

		return true;
	}
};
int Dataset::nFeatures = 0;
int Dataset::nClasses = 0;
set<int>* Dataset::classifications = nullptr;

struct Cluster
{
	vector<double> center;		// center of the cluster
	set<Dataset*> assigned; 	// training examples belonging to the cluster
	int classification;			// classification of the cluster; determined when clusters settled

	// create a cluster at point p
	Cluster(vector<double> p) : center(p), classification(-1) {}

	// set classification to majority classification of assigned training points
	void calculateClassification()
	{
		map<int, int> classCounts;
		for (auto it = Dataset::classifications->begin(); it != Dataset::classifications->end(); it++)
			classCounts[*it] = 0;

		for (Dataset* ds : assigned)
			classCounts.find(ds->classification)->second++;

		auto majorityClass = classCounts.begin();
		for (auto it = classCounts.begin(); it != classCounts.end(); it++)
			if (it->second > majorityClass->second)
				majorityClass = it;

		classification = majorityClass->first;
	}

	// return the distance from the center of the cluster to a point
	double distanceTo(vector<double> p)
	{
		double d = 0;

		for (int i = 0; i < Dataset::nFeatures; i++)
			d += pow(p[i] - center[i], 2);

		return sqrt(d);
	}

	// generate a new center by averaging the components of each assigned training point
	vector<double> generateCenter()
	{
		vector<double> newCenter;
		for (int i = 0; i < Dataset::nFeatures; i++)
			newCenter.push_back(0.0);

		for (Dataset* ds : assigned)
			for (int i = 0; i < Dataset::nFeatures; i++)
				newCenter[i] += ds->features[i];

		for (double& d : newCenter)
			d /= assigned.size();

		return newCenter;
	}	

	// assign a dataset to the cluster by adding it to the set of assigned datasets
	// return whether or not the dataset was successfully inserted
	bool assign(Dataset* ds)
	{
		return assigned.insert(ds).second;
	}
};

// just using a class here since it seems more organized
class KMeans
{
public:
	static int nClusters;
	
	KMeans(vector<Dataset*> t) : training(t)
	{
		initClusters();

		#ifdef DBG
		cout << "Initalized clusters: " << endl;
		for (Cluster* c : clusters)
		{
			cout << "\tCenter: ";
			for (double d : c->center)
				cout << d << " ";
			cout << endl;
		}
		cout << endl;
		#endif

		// main kmeans loop
		do
		{
			// for each training example
			for (Dataset* ds : training)
			{
				// map the address of the closest cluster to the address of the training example
				// and assign the training example to its mapped cluster
				dataClusterMap[ds] = clusterClosestTo(ds->features);
				dataClusterMap[ds]->assign(ds);
			}

			// for each cluster centroid
			for (int i = 0; i < nClusters; i++)
			{
				// update the location to the average location of all currently assigned training examples
				// and clear the clear the cluster's vector of assigned training examples
				//
				// Now each key (address of a training example) in dataClusterMap will point at the
				// cluster to which the respective training example was previously assigned
				clusters[i]->center = clusters[i]->generateCenter();
				clusters[i]->assigned.clear();
			}
		}
		while (!clustersSettled());

		// now that clusters are settled, we need to perform a final assignment of training examples
		// to their nearest cluster centroid
		for (Dataset* ds : training)
			clusterClosestTo(ds->features)->assign(ds);

		// each cluster will have a label corresponding to the majority label of its assigned training
		// examples
		for (Cluster* c : clusters)
			c->calculateClassification();

		#ifdef DBG
		cout << "Clusters settled with the following data:" << endl;
		for (Cluster* c : clusters)
		{
			cout << "\tCenter: ";
			for (double d : c->center) cout << d << " ";
			cout << endl;
			cout << "\tClassification: " << c->classification << endl;
			cout << "\tAssigned points: " << c->assigned.size() << endl;
			cout << endl;
		}
		#endif
	}

	// attempt to classify a dataset and return the integer label of the classification
	int classify(Dataset* ds)
	{
		return clusterClosestTo(ds->features)->classification;
	}

private:
	vector<Cluster*> clusters;
	vector<Dataset*> training;
	map<Dataset*, Cluster*> dataClusterMap; // maps each set of data to a cluster

	// initalize clusters as random unique training points
	void initClusters()
	{
		#ifdef DBG
		cout << "Initializing clusters . . ." << endl;
		#endif

		set<int> randomIndices;
		
		for (int i = rand() % training.size(); clusters.size() != nClusters; i = rand() % training.size())
			if (randomIndices.insert(i).second)
				clusters.push_back(new Cluster(training[i]->features));

		#ifdef DBG
		cout << "Clusters initialized." << endl;
		#endif
	}

	// given a vector of clusters cVec, return the one that's closest to a given point p
	Cluster* clusterClosestTo(vector<double> p)
	{
		Cluster* closest = clusters[0];

		for (Cluster* c : clusters)
			if (c->distanceTo(p) < closest->distanceTo(p))
				closest = c;

		return closest;
	}

	// clusters are considered settled if all points are assigned to the same cluster as on a 
	// previous iteration
	bool clustersSettled()
	{
		for (Dataset* ds : training)
			if (dataClusterMap[ds] != clusterClosestTo(ds->features))
				return false;

		return true;
	}
};
int KMeans::nClusters = 0;

int main(int argc, char** argv)
{
	if (argc != 6)
	{
		cout << "Program takes 5 arguments:" << endl;
		cout << "\t1) int: random seed" << endl;
		cout << "\t2) int: number of clusters" << endl;
		cout << "\t3) int: number of features in data set" << endl;
		cout << "\t4) string: training data filename" << endl;
		cout << "\t5) string: testing data filename" << endl;
		return -1;
	}

	int seed 			= atoi(argv[1]);
	KMeans::nClusters	= atoi(argv[2]);
	Dataset::nFeatures	= atoi(argv[3]);
	ifstream trainingFile;
	ifstream testingFile;
	vector<Dataset*> trainingData;
	vector<Dataset*> testingData;

	// init rand
	srand(seed);

	// open files for reading
	trainingFile.open(argv[4]);
	testingFile.open(argv[5]);

	// read in the training data
	string dataline;
	getline(trainingFile, dataline);
	while (!trainingFile.eof())
	{
		trainingData.push_back(new Dataset(dataline));
		getline(trainingFile, dataline);
	}

	// set number of classifications we're working with
	set<int> classifications;
	for (Dataset* ds : trainingData)
		if (classifications.insert(ds->classification).second)
			Dataset::nClasses++;
	Dataset::classifications = &classifications;

	// set up kmeans data and perform kmeans clustering
	KMeans km(trainingData);

	// read in the testing data
	getline(testingFile, dataline);
	while (!testingFile.eof())
	{
		testingData.push_back(new Dataset(dataline));
		getline(testingFile, dataline);
	}

	// classify each dataset and print the number correct
	int nCorrect = 0;
	for (Dataset* ds : testingData)
		if (km.classify(ds) == ds->classification)
			nCorrect++;

	// output number of test data correctly identified
	#ifdef DBG
	cout << "Number testing examples correctly classified: ";
	#endif

	cout << nCorrect << endl;

	// close files
	trainingFile.close();
	testingFile.close();

	return 0;
}
