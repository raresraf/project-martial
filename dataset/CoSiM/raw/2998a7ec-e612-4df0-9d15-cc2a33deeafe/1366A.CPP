#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
  ll t;
  cin >> t;
  while (t--) {
    ll a, b;
    cin >> a >> b;
    cout << min((a + b) / 3, min(a, b)) << "\n";
  }
}