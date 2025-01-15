#include <bits/stdc++.h>
typedef long long int ll;
using namespace std;
int main() {
  ll n, m;
  cin >> n >> m;
  ll area = n * m;
  ll j = 2;
  if (area < j) {
    cout << "0" << endl;
  } else {
    ll k = n * m;
    k = k / 2;
    cout << k << endl;
  }
  return 0;
}