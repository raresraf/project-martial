// 4A.cpp
// A. Watermelon
// Author: derekhh
// May 31, 2015

#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  if (w % 2 == 0 && w != 2)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}