///   ***   ---   In the name of ALLAH    ---   ***   ///
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<int> VI;
typedef vector<ll> VL;
typedef vector<VI> VVI;
typedef vector<VL> VVL;
typedef pair<int, int> PII;
typedef pair<double, double> PDD;
typedef pair<ll, ll> PLL;
typedef vector<PII> VII;
typedef vector<PLL> VLL;
typedef double DL;
#define endl '\n'
#define PB push_back
#define F first
#define S second
#define all(a) (a).begin(), (a).end()
#define rall(a) (a).rbegin(), (a).rend()
#define print cout
#define sz(x) (int)x.size()
const double PI = acos(-1);
const double eps = 1e-9;
const int inf = 2000000000;
const ll infLL = 9000000000000000000;
#define MOD 1000000007
#define mem(a, b) memset(a, b, sizeof(a))
#define sqr(a) ((a) * (a))
#define fori(n) for (ll i = 0; i < n; ++i)
#define forj(n) for (ll j = 0; j < n; ++j)
#define forv(v) for (auto i : v)
#define minimum(a) *min_element(a.begin(), a.end())
#define maximum(a) *max_element(a.begin(), a.end())
#define optimize()                                                             \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define fraction()                                                             \
  cout.unsetf(ios::floatfield);                                                \
  cout.precision(10);                                                          \
  cout.setf(ios::fixed, ios::floatfield);
#define file() freopen("input.txt", "r", stdin);
#define dbg(args...)                                                           \
  do {                                                                         \
    cerr << #args << " : ";                                                    \
    faltu(args);                                                               \
  } while (0)
void faltu() { cerr << endl; }
template <typename T, typename... hello>
void faltu(T arg, const hello &...rest) {
  cerr << arg << ' ';
  faltu(rest...);
}
ll gcd(ll a, ll b) { return __gcd(a, b); }
ll lcm(ll a, ll b) { return a * (b / gcd(a, b)); }
ll max(ll a, ll b) { return a > b ? a : b; }
ll min(ll a, ll b) { return a < b ? a : b; }
int main() {
  optimize();
  string s;
  cin >> s;
  int cnt = 0;
  for (int i = 0; i < sz(s) - 1; i++) {
    if (s[i] == s[i + 1]) {
      cnt++;
    } else
      cnt = 0;
    if (cnt >= 6) {
      print << "YES\n";
      return 0;
    }
  }
  print << "NO\n";
  return 0;
}
/*
*****Comment Section**********



*/
