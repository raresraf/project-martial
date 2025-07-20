#include <bits/stdc++.h>
typedef long long ll;
using namespace std;

int main() {
  ll t;
  scanf("%lld", &t);
  while (t--) {
    ll a, b, c;
    scanf("%lld %lld %lld", &a, &b, &c);
    ll temp = min(min(a, b), (a + b + c) / 3);
    printf("%lld\n", temp);
  }
}
