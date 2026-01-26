// 282A.cpp
// A. Bit++
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, x = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string str;
    cin >> str;
    if (str.find('+') != string::npos)
      x++;
    else
      x--;
  }
  cout << x << endl;
  return 0;
}