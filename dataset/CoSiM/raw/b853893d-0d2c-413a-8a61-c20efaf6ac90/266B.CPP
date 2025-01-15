#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  string s;
  cin >> s;
  for (int l = 0; l < k; l++) {
    for (int i = 1; i < n; ++i) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i] = 'B';
        s[i - 1] = 'G';
        ++i;
      }
    }
  }
  cout << s;
  return 0;
}