#include <bits/stdc++.h>
#define ll long long
#define pb push_back
#define mkp make_pair
#define pii pair<int, int>
using namespace std;
vector<ll> vec;
int main() {
  int t;
  scanf("%d", &t);
  ll n;
  ll vs, sum;
  while (t--) {
    scanf("%lld", &n);
    ll ar[n + 1];
    for (ll i = 0; i < n; i++) {
      scanf("%lld", &ar[i]);
    }
    vec.clear();
    vec.pb(ar[0]);
    vs = 1;
    for (ll i = 1; i < n; i++) {
      if ((ar[i] > 0 && vec[vs - 1] < 0) || (ar[i] < 0 && vec[vs - 1] > 0)) {
        vec.pb(ar[i]);
        vs++;
      } else if ((ar[i] < 0 && vec[vs - 1] < 0) ||
                 (ar[i] > 0 && vec[vs - 1] > 0)) {
        if (ar[i] > vec[vs - 1]) {
          vec[vs - 1] = ar[i];
        }
      }
    }

    sum = 0;
    for (ll i = 0; i < vs; i++) {
      sum += vec[i];
    }
    printf("%lld\n", sum);
  }
  return 0;
}
