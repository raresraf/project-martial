#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;
  int f = 0;
  for (int i = 0; i < a.length(); i++) {
    if (tolower(a[i]) > tolower(b[i])) {
      cout << 1;
      f = 1;
      break;
    } else if (tolower(a[i]) < tolower(b[i])) {
      cout << -1;
      f = 1;
      break;
    }
  }
  if (f == 0)
    cout << 0;
}
