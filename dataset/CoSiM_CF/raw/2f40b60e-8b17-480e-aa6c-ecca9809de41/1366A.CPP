#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define pf push_front
#define ll long long

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int t, a, b, c;
  cin >> t;
  while (t--) {
    cin >> a >> b;
    c = (a + b) / 3;
    cout << min(a, min(b, c)) << "\n";
  }

  return 0;
}