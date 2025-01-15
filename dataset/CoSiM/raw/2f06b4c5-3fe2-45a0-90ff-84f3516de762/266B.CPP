// 266B.cpp
// Queue at the School
// Author: Ikki
// Jun 11, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string a, b;
  cin >> a;
  b = a;
  for (int tt = 0; tt < t; tt++) {
    for (int i = 0; i < n - 1;) {
      if (a[i] == 'B' && a[i + 1] == 'G') {
        b[i] = 'G';
        b[i + 1] = 'B';
        i += 2;
      } else {
        b[i] = a[i];
        i++;
      }
    }
    a = b;
  }
  cout << b << endl;
  return 0;
}