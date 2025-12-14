//============================================================================
// Name        : 110A.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int main() {
  long long x;
  cin >> x;
  int c = 0;
  while (x) {
    int d = x % 10;
    x = x / 10;
    if (d == 4 || d == 7)
      c++;
  }
  if (c == 4 || c == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
