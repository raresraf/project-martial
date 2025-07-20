#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, i, c, s = 0;
  char a[5];
  string b;
  cin >> b;
  a[0] = 'h';
  a[1] = 'e';
  a[2] = 'l';
  a[3] = 'l';
  a[4] = 'o';
  for (i = 0; i < b.size(); i++) {
    if (b[i] == a[s]) {
      s++;
    }
  }
  if (s >= 5) {
    cout << "YES";
  } else {
    cout << "NO" << endl;
  }
  return 0;
}