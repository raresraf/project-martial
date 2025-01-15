#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
typedef long double ld;
ll gcd(ll a, ll b) {
  if (a == 0 || b == 0)
    return 0;

  if (a == b)
    return a;

  if (a > b)
    return gcd(a - b, b);
  return gcd(a, b - a);
}

int main() {
  ll n, a, b, temp, i, counta = 0, countb = 0;
  cin >> n;
  cin >> a >> b;
  if (a > b) {
    while (n % b != 0 && n > 0) {
      n -= a;
      counta++;
    }
    if (n < 0) {
      cout << "NO" << endl;
      return 0;
    }
    cout << "YES" << endl;
    cout << counta << " " << n / b << endl;
    return 0;
  }
  while (n % a != 0 && n > 0) {
    n -= b;
    countb++;
  }
  if (n < 0) {
    cout << "NO" << endl;
    return 0;
  }
  cout << "YES" << endl;
  cout << n / a << " " << countb << endl;
  return 0;

  return 0;
}