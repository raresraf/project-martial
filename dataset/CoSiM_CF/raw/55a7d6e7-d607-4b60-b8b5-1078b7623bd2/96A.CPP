#include <bits/stdc++.h>
using namespace std;

int main() {
  int b, e = 0, g, i, j, d;
  string a;
  cin >> a;
  for (i = 0; i < a.size(); i++) {
    g = 1;
    for (j = i + 1; j < a.size(); j++) {
      if (a[i] == a[j]) {
        g = g + 1;
        if (g == 7) {
          cout << "YES";
          e = e + 1;
          break;
        }
      } else {
        break;
      }
    }
    if (g == 7) {
      break;
    }
  }
  if (e == 0) {
    cout << "NO";
  }
}