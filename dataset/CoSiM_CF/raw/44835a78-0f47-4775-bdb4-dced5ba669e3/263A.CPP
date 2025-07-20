// 263A.cpp
// A. Beautiful Matrix
// Author: Ikki
// Jun 11, 2015

#include <cstdlib>
#include <iostream>
using namespace std;

int gr[5][5];

int main() {
  int r, c;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> gr[i][j];
      if (gr[i][j] == 1) {
        r = i;
        c = j;
      }
    }
  }
  cout << abs(2 - r) + abs(2 - c) << endl;
  return 0;
}