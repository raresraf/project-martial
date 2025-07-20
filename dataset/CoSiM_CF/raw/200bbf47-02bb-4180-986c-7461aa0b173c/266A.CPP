#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, n, c = 0;
  cin >> n;
  string s;
  cin >> s;
  for (i = 0; s[i] != '\0'; i++) {
    if (s[i] == s[i + 1])
      c++;
  }

  cout << c;
}
