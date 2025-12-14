#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, t;
  cin >> s;
  t = "hello";
  if (s.length() < 5) {
    cout << "NO" << endl;
    return 0;
  }

  int i = 0, j = 0;
  while (i < s.length() && j < 5) {
    if (s[i] == t[j]) {
      i++;
      j++;
    } else {
      i++;
    }
  }
  if (j >= 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}