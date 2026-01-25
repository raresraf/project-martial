#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
int main() {
  ll n;
  cin >> n;
  ll k = n, r, c = 0;
  while (k > 0) {
    r = k % 10;
    if (r == 4 || r == 7)
      c++;
    k /= 10;
  }
  if (c == 4 || c == 7)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}