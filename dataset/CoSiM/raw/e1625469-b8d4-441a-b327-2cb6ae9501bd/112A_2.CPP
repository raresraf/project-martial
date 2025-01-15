#include <bits/stdc++.h>
using namespace std;

void yoyo(string &a) {
  for (int i = 0; i < a.length(); i++) {
    if (a[i] >= 'A' && a[i] <= 'Z') {
      char ch = a[i] + 'a' - 'A';
      a[i] = ch;
    }
  }
}

int main() {
  string a, b;
  cin >> a >> b;
  yoyo(a);
  yoyo(b);
  for (int i = 0; i < a.length(); i++) {
    if (a[i] > b[i]) {
      cout << 1;
      return 0;
    }
    if (a[i] < b[i]) {
      cout << -1;
      return 0;
    }
  }
  cout << 0;
}