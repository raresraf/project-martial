#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  int i, j;
  cin >> a;
  j = a[0];
  if (j >= 97) {
    j = j - 32;
  }
  a[0] = j;
  for (i = 0; i < a.size(); i++) {
    cout << a[i];
  }
}