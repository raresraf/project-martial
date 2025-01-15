#include <bits/stdc++.h>
#include <iostream>
#include <math.h>
using namespace std;

int main() {
  int n, temp;
  cin >> n;
  temp = n;
  while (n--) {
    string a;
    cin >> a;
    int l = a.length();
    if (l > 10) {
      int lc = l - 2;
      cout << a.at(0) << lc << a.at(l - 1) << endl;
    } else {
      cout << a << endl;
    }
  }
  return 0;
}
