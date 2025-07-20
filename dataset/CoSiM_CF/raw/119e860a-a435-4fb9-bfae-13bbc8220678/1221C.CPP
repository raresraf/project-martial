#include <bits/stdc++.h>

using namespace std;

#define INF 1000000000
#define INFLL 0x3f3f3f3f3f3f3f3fLL
#define EPS 10e-9
#define MOD 1000000007
#define mp make_pair
#define mt make_tuple
#define pb push_back
#define st first
#define nd second
#define sz(v) int(v.size())
#define all(X) (X).begin(), (X).end()
#define FOR(I, A, B) for (int I = A; I < B; I++)
#define RFOR(I, A, B) for (int I = A; I >= B; I--)

typedef long long ll;
typedef pair<int, int> ii;
typedef pair<int, ii> iii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ii> vii;
typedef vector<vii> vvii;
typedef vector<iii> viii;
typedef vector<ll> vll;

bool possible(ll c, ll m, ll x, ll n) {
  if (n > c)
    return false;
  if (n > m)
    return false;
  c -= n;
  m -= n;
  if (c + m + x >= n)
    return true;
  return false;
}

int main() {
  int q;
  scanf("%d", &q);
  while (q--) {
    ll c, m, x;
    scanf("%lld %lld %lld", &c, &m, &x);
    ll l = 0, r = 100000010;
    while (l < r - 1) {
      // cout << l << " " << r << endl;
      ll mid = (l + r) / 2;
      bool pos = possible(c, m, x, mid);
      if (pos) {
        l = mid;
      } else {
        r = mid;
      }
    }
    printf("%lld\n", l);
  }
  return 0;
}