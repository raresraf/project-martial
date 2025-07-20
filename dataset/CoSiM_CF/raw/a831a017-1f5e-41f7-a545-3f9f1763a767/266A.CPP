#include <bits/stdc++.h>
#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int a = 0, b = 0, c = 0, r = 0, n;
  cin >> n;
  string s;
  cin >> s;
  for (int i = 0; i < n; i++) {
    if (s[i] == 'R') {
      a++;
      b = 0;
      c = 0;
    }
    if (s[i] == 'G') {
      b++;
      a = 0;
      c = 0;
    }
    if (s[i] == 'B') {
      c++;
      b = 0;
      a = 0;
    }
    if (a == 2) {
      a = 1;
      r++;
    }
    if (c == 2) {
      c = 1;
      r++;
    }
    if (b == 2) {
      b = 1;
      r++;
    }
  }
  cout << r << endl;
}