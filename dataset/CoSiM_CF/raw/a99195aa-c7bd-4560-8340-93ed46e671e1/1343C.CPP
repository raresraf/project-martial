#include <bits/stdc++.h>
#define ll long long
using namespace std;

bool dif(ll a, ll b) { return abs(abs(a) - abs(b)) != abs(a - b); }

int main() {
  int t;
  scanf(" %d", &t);
  while (t--) {
    int n;
    scanf(" %d", &n);
    ll last = 0;
    ll ans = 0;
    for (int i = 0; i < n; ++i) {
      ll va;
      scanf(" %lld", &va);
      if (dif(last, va) || i == 0)
        ans += last, last = va;
      else
        last = max(last, va);
    }
    printf("%lld\n", ans + last);
  }
  return 0;
}
