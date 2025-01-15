#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <math.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  char a[100];
  int j = 0;
  ;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] != '+') {
      a[j] = s[i];
      j++;
    }
  }
  sort(a, a + j);
  if (j == 1)
    cout << a[0];
  else {
    for (int i = 0; i < j; i++) {
      if (i == j - 1)
        cout << a[i];
      else {
        cout << a[i];
        cout << "+";
      }
    }
  }
}
