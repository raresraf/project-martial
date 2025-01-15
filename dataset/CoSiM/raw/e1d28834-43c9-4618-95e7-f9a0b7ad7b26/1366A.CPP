#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  int t;
  scanf("%d", &t);
  while (t--) {
    ll a, b;
    scanf("%lld %lld", &a, &b);

    ll res = 0;
    if (a > b) {
      ll diff = a - b;
      diff = min(b, diff);
      res += diff;
      a -= 2 * diff;
      b -= diff;
    } else if (b > a) {
      ll diff = b - a;
      diff = min(a, diff);
      res += diff;
      a -= diff;
      b -= 2 * diff;
    }
    ll l = min(a, b);
    ll amt = l / 3;
    res += amt * 2;
    a -= amt * 3;
    b -= amt * 3;
    if (a && b && a + b >= 3)
      res++;
    printf("%lld\n", res);
  }
}
