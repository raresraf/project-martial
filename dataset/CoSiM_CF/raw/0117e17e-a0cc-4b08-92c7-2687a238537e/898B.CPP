#include <bits/stdc++.h>
#define ll long long int
using namespace std;
int main() {
  ll n, a, b;
  cin >> n >> a >> b;

  for (ll i = 0; i <= n; i++) {
    if ((n - (a * i)) >= 0 && (n - (a * i)) % b == 0) {
      cout << "YES\n" << i << " " << (n - (a * i)) / b << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}
