#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t, a;
  cin >> n >> t;
  string s;
  cin >> s;

  int i, j, k;

  for (i = 0; i < t; i++) {
    for (j = 1; j < n; j++) {
      if (s[j - 1] == 'B' && s[j] == 'G') {
        swap(s[j - 1], s[j]);
        j++;
      }
    }
  }
  cout << s;

  return 0;
}