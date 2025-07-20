#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, hello;
  hello = "hello";
  int j = 0;
  cin >> s;
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] == hello[j]) {
      j += 1;
      if (j == 5) {
        cout << "YES\n";
        break;
      }
    }
  }
  if (j != 5) {
    cout << "NO\n";
  }

  return 0;
}