#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

typedef long long ll;

int main() {
  ll n, c = 0;

  for (cin >> n; n > 0; n /= 10) {
    ll d = n % 10;
    c += d == 4 || d == 7;
  }

  if (c == 4 || c == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
