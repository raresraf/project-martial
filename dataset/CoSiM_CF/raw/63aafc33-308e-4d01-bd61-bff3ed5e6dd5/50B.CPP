#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
#define ten5 100000 + 10
#define MOD 1000000007
#define rep(i, a, n) for (int i = a; i < n; i++)
#define iif(c, t, f) ((c) ? (t) : (f))
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
#define mp make_pair

int intcmp(const void *v1, const void *v2) { return *(int *)v1 - *(int *)v2; }

int main() {
  int q;
  cin >> q;
  while (q-- > 0) {
    ll n, m, k;
    scanf("%lld %lld %lld", &n, &m, &k);
    ll minstep = max(n, m);
    if (k < minstep) {
      printf("-1\n");
      continue;
    }

    if ((minstep - min(m, n)) % 2 == 1) {
      printf("%lld\n", k - 1);
    } else if ((k - minstep) % 2 == 1) {
      printf("%lld\n", k - 2);
    } else {
      printf("%lld\n", k);
    }
  }
  return 0;
}
