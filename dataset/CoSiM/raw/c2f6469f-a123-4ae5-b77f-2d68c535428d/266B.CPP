#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n, t, i;
  string s;
  cin >> n >> t >> s;
  int ind[n];
  for (i = 0; i < n - 1; i++) {
    ind[i] = 0;
  }
  while (t--) {
    for (i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        ind[i] = 1;
      }
    }
    for (i = 0; i < n - 1; i++) {
      if (ind[i] == 1) {
        swap(s[i], s[i + 1]);
      }
    }
    for (i = 0; i < n - 1; i++) {
      ind[i] = 0;
    }
  }
  cout << s << endl;
  return 0;
}