#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string c;
  long long lengths, n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> c;
    lengths = c.length();
    if (lengths > 10) {
      cout << c[0] << lengths - 2 << c[lengths - 1] << "\n";
    } else {
      cout << c << "\n";
    }
  }

  return 0;
}
