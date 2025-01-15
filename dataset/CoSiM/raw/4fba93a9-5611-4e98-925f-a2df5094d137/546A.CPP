#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int k, w;
  ll n;
  cin >> k >> n >> w;
  ll t = 0;
  for (int i = 1; i <= w; ++i) {
    t += k * i;
  }
  t - n >= 0 ? cout << t - n : cout << 0;

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }