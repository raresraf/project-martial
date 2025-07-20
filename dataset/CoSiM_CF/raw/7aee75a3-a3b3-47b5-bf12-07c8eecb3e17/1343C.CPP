#include <bits/stdc++.h>

#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define ER(x) cerr << #x << " = " << x << endl;
#define ERR(x, y) cerr << #x << " = " << x << " , " << #y << " = " << y << endl;
#define ERRR(x, y, z)                                                          \
  cerr << #x << " = " << x << " , " << #y << " = " << y << " , " << #z         \
       << " = " << z << endl;
#define F first
#define S second
#define pb push_back
#define LOG(x)                                                                 \
  for (int i = 0; i <= n; i++) {                                               \
    cout << #x << "[" << i << "] = " << x[i] << endl;                          \
  }
#define LOG2(x)                                                                \
  for (int i = 0; i <= n; i++) {                                               \
    for (int j = 0; j <= n; j++) {                                             \
      cout << #x << "[" << i << "][" << j << "] = " << x[i][j] << endl;        \
    }                                                                          \
  }
#define fore(i, l, r) for (ll i = l; i <= r; i++)
#define forn(i, l, r) for (int i = l; i >= r; i--)

using namespace std;
typedef long long ll;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;

const int INF32 = 1e9;
const ll INF64 = 1e16;
const ll MOD = 1e9 + 7;
const ll maxn = 2e5 + 7;
const ll maxm = 2e5 + 7;

ll t, n, m, a[maxn];

int main() {
  fastIO;
  cin >> t;
  while (t--) {
    ll firstManfi = -1, firstMosbat = -1;
    cin >> n;
    fore(i, 1, n) {
      cin >> a[i];
      if (firstManfi == -1 && a[i] < 0) {
        firstManfi = i;
      }
      if (firstMosbat == -1 && a[i] > 0) {
        firstMosbat = i;
      }
    }
    ll last, dpFirstManfi, cntFirstManfi = 0, dpFirstMosbat, cntFirstMosbat = 0;
    if (firstManfi > 0) {
      last = a[firstManfi];
      cntFirstManfi = 1;
      dpFirstManfi = last;
      fore(i, firstManfi + 1, n) {
        if (last < 0) {
          if (a[i] > 0) {
            dpFirstManfi += a[i];
            cntFirstManfi++;
            last = a[i];
          } else if (a[i] < 0 && a[i] > last) {
            dpFirstManfi += a[i] - last;
            last = a[i];
          }
        } else {
          if (a[i] < 0) {
            dpFirstManfi += a[i];
            cntFirstManfi++;
            last = a[i];
          } else if (a[i] > 0 && a[i] > last) {
            dpFirstManfi += a[i] - last;
            last = a[i];
          }
        }
      }
    }
    if (firstMosbat > 0) {
      cntFirstMosbat = 1;
      last = a[firstMosbat];
      dpFirstMosbat = last;
      fore(i, firstMosbat + 1, n) {
        if (last < 0) {
          if (a[i] > 0) {
            cntFirstMosbat++;
            dpFirstMosbat += a[i];
            last = a[i];
          } else if (a[i] < 0 && a[i] > last) {
            dpFirstMosbat += a[i] - last;
            last = a[i];
          }
        } else {
          if (a[i] < 0) {
            cntFirstMosbat++;
            dpFirstMosbat += a[i];
            last = a[i];
          } else if (a[i] > 0 && a[i] > last) {
            dpFirstMosbat += a[i] - last;
            last = a[i];
          }
        }
      }
    }

    if (cntFirstMosbat > cntFirstManfi) {
      cout << dpFirstMosbat << endl;
    } else if (cntFirstMosbat < cntFirstManfi) {
      cout << dpFirstManfi << endl;
    } else {
      cout << max(dpFirstMosbat, dpFirstManfi) << endl;
    }
  }

  return 0;
}