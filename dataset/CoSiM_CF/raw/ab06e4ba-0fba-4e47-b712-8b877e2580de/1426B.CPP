// m1 m2
// m3 m4

#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  short t, n, m, m1, m2, m3, m4;
  cin >> t;
  while (t--) {
    cin >> n >> m;
    bool possible = false;

    while (n--) {
      cin >> m1 >> m2 >> m3 >> m4;
      if (m2 == m3)
        possible = true;
    }

    if (m % 2)
      possible = false;

    possible ? cout << "YES\n" : cout << "NO\n";
  }
}