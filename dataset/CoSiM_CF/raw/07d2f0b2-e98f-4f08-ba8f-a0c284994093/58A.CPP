#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  string h = "hello";

  cin >> s;

  int j = 0;

  for (int i = 0; i < (int)s.length(); ++i) {
    if (s[i] == h[j]) {
      j++;
    }
  }

  if (j == 5) {
    cout << "YES";
  } else {
    cout << "NO\n";
  }
}
