#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <cmath>
using namespace std;
#include "common.h"
#include "basis.h"
#include "hmatrix.h"

int main() {

	// model information
	cout << "Full exact Diadonalization  for many-body localization problems" << endl << endl;

	int time0, time1, time2, time3, time4;
	time0 = time(0);
	// find states

	Hmatrix hmatrix;

	cout << endl;
	cout << "E_min: " << hmatrix.spec[0] << endl;
	cout << "E_max: " << hmatrix.spec[hmatrix.Dim-1] << endl;

	return 0;
}
