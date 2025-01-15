#include <bits/stdc++.h>
using namespace std;
#define fst first
#define snd second
typedef unsigned long long ull;
typedef long long ll;
typedef pair<int, int> pii;
#define pb push_back
#define for_tests(t, tt)                                                       \
  int t;                                                                       \
  scanf("%d", &t);                                                             \
  for (int tt = 1; tt <= t; tt++)
#ifndef ONLINE_JUDGE
#define debug(args...) fprintf(stderr, args)
#else
#define debug(args...)
#endif // ONLINE_JUDGE
template <typename T> inline T abs(T t) { return t < 0 ? -t : t; }
const ll modn = 1000000007;
inline ll mod(ll x) { return x % modn; }

const int MAXN = 212345;

ll n, m, k;

int main() {
  for_tests(t, tt) {
    scanf("%lld%lld", &n, &m);
    if (n < m)
      swap(n, m);
    ll res = min(n / 2, m);
    ll ans = 0;
    ll dif = n - m;
    if (dif <= m) {
      n -= 2 * dif;
      m -= dif;
      ans += dif;
      assert(n == m);
      // printf("n %lld  m %lld\n", n,m);
      ans += (n / 3) * 2;
      if (n % 3 == 2)
        ans++;
    }
    // printf("ans %lld res %lld\n",ans, res);
    printf("%lld\n", max(ans, res));
  }
}
