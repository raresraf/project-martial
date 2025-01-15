#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, t, i;
  cin >> n >> t;
  string q;
  cin >> q;

  while (t--) {
    for (i = 0; i < n; i++) {
      if (q[i] == 'B' && q[i + 1] == 'G') {
        q[i] = 'G';
        q[i + 1] = 'B';
        i++;
      }
    }
  }

  cout << q;
}