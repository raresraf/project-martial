#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  string s;
  cin >> s;
  int z = 0;
  int u = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '0') {
      z += 1;
      u = 0;
    } else {
      z = 0;
      u += 1;
    }
    if (z >= 7 || u >= 7) {
      cout << "YES"
           << "\n";
      return 0;
    }
  }
  cout << "NO"
       << "\n";
  return 0;
}
