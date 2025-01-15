#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  string in1, in2;

  cin >> in1 >> in2;

  transform(in1.begin(), in1.end(), in1.begin(), ::tolower);
  transform(in2.begin(), in2.end(), in2.begin(), ::tolower);

  cout << (in1 > in2 ? 1 : (in1 == in2 ? 0 : -1)) << endl;

  return 0;
}
