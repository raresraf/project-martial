#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;

  for (int i = 0; i < s.length(); i++) {
    if (!isupper(s[0]))
      s[0] -= 32;
  }
  cout << s;
}
