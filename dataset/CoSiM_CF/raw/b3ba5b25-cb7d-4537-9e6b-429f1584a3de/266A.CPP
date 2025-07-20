#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n, c(0);
  cin >> n;
  string s;
  cin >> s;

  for (int i = 0; i < n; i++) {
    if (s[i] == s[i + 1])
      c++;
  }

  cout << c << "\n";
}
