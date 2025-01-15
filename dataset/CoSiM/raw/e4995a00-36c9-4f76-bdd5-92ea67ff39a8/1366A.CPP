#include <bits/stdc++.h>
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define ll long long
using namespace std;

ll t, a, b;

int main() {
  cin >> t;
  while (t--) {
    cin >> a >> b;
    cout << min((a + b) / 3, min(a, b)) << '\n';
  }
  return 0;
}