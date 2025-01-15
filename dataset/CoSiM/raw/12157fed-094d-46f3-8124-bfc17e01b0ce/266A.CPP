#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int i, n, c;

  cin >> n;
  c = 0;
  cin >> s;
  for (i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1])
      c++;
  }
  cout << c << endl;

  return 0;
}
