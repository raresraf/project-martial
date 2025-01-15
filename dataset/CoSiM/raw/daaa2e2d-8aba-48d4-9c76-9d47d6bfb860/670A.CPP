#include <bits/stdc++.h>
#define ll long long int
using namespace std;

void solve() {
  ll n;
  cin >> n;

  ll mini, maxx;

  if (n % 7 == 6)
    mini = 2 * (n / 7) + 1;
  else
    mini = 2 * (n / 7);
  if (n % 7 == 1) {
    maxx = mini + 1;
  } else if (n % 7 > 1) {
    if (n % 7 == 6) {
      maxx = mini + 1;
    } else
      maxx = mini + 2;
  } else
    maxx = mini;

  cout << mini << " " << maxx;
}

int main() {
  solve();
  return 0;
}
