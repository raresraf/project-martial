/*
 ______   _________ _      _       _
|  ____ \|___   ___| |    | |     | |
| |    \ \   | |   | |____| |     | |
| |    | |   | |   |  ____  |     | |
| |____/ /___| |___| |    | |_____| |
|_______/|_________|_|    |_________|

*/
#include <bits/stdc++.h>
using namespace std;
#define ed end()
#define INF 1e18
#define fi first
#define sz size()
#define se second
#define flt float
#define bg begin()
#define str string
#define eif else if
#define si set<int>
#define sll set<ll>
#define dbl double
#define len length()
#define ll long long
#define pb push_back
#define pf push_front
#define mt make_tuple
#define pi 3.14159265
#define spii set<pii>
#define mll map<ll, ll>
#define vll vector<ll>
#define vi vector<int>
#define vvi vector<vi>
#define MOD 1000000007
#define p_b pop_back()
#define p_f pop_front()
#define u_b upper_bound
#define l_b lower_bound
#define pll pair<ll, ll>
#define ui unsigned int
#define vull vector<ull>
#define bit(a) bitset<a>
#define vpii vector<pii>
#define vpll vector<pll>
#define mii map<int, int>
#define mii map<int, int>
#define mci map<char, int>
#define pii pair<int, int>
#define mci map<char, int>
#define vd vector<double>
#define vs vector<string>
#define u_s unordered_set
#define u_m unordered_map
#define b_s binary_search
#define ti(x, i) get<i>(x)
#define vtiii vector<tiii>
#define max_of max_element
#define min_of min_element
#define gcd(a, b) __gcd(a, b)
#define pstv(x)                                                                \
  if (x < 0)                                                                   \
  x = 0
#define ngtv(x)                                                                \
  if (x > 0)                                                                   \
  x = 0
#define out(a) cout << a << "\n"
#define outn(a) cout << a << endl
#define lcm(a, b) a *b / gcd(a, b)
#define read(x) for (auto &i : x)
#define visit(x) for (auto i : x)
#define usi unordered_set<int>
#define ull unsigned long long
#define pqi priority_queue<int>
#define tiii tuple<int, int, int>
#define ptr(x) x.begin(), x.end()
#define mx(a, b, c) max(a, max(b, c))
#define mn(a, b, c) min(a, min(b, c))
#define srtd(x) is_sorted(ptr(x))
#define str_in(x)                                                              \
  string x;                                                                    \
  cin >> x
#define rptr(x) x.rbegin(), x.rend()
#define umii unordered_map<int, int>
#define umci unordered_map<char, int>
#define _for(i, a, n) for (i = a; i < n; i++)
#define for_(i, a, n) for (i = a; i > n; i--)
#define maxof(x) max_element(ptr(x))
#define minof(x) min_element(ptr(x))
#define srt(x) sort(x.begin(), x.end())
#define out2(a, b) cout << a << " " << b << endl;
#define _pow(a, b, c)                                                          \
  ull c = 1;                                                                   \
  for (; b--; c *= a)
#define vs_in(x, n)                                                            \
  vs x(n);                                                                     \
  read(x) cin >> i
#define vs_out(x, n) visit(x) cout << i << endl
#define sum(x, s)                                                              \
  ll s = 0;                                                                    \
  for (auto i : x)                                                             \
  s += i
#define adjdif(x) for_(i, n - 1, 0) x[i] -= x[i - 1]
#define ub(x, a) upper_bound(ptr(x), a) - x.begin()
#define lb(x, a) lower_bound(ptr(x), a) - x.begin()
#define rotate_l(x, a) rotate(x.bg, x.bg + a, y.ed)
#define v_in(x, n)                                                             \
  vll x(n);                                                                    \
  for (auto &i : x)                                                            \
  cin >> i
#define bs(x, a) binary_search(ptr(x), a) - x.begin()
#define pre_sum(x) _for(i, 1, x.size()) x[i] += x[i - 1]
#define rotate_r(x, a) rotate(x.bg, x.bg + x.sz - a, y.ed)
#define FOR(i, x) for (auto i = x.begin(); i != x.end(); i++)
#define d_sum(n, s)                                                            \
  ll s = 0, a = n;                                                             \
  while (a)                                                                    \
  s += a % 10, a /= 10
#define v_out(x)                                                               \
  for (auto i : x)                                                             \
    cout << i << " ";                                                          \
  cout << "\n"
#define v_outn(x)                                                              \
  for (auto i : x)                                                             \
    cout << i << " ";                                                          \
  cout << endl
#define ____                                                                   \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(nullptr);
#define vp_in(x, n)                                                            \
  vpll x(n);                                                                   \
  for (auto &i : x)                                                            \
  cin >> x.fi >> x.se
#define rev(n)                                                                 \
  ull rev;                                                                     \
  while (n) {                                                                  \
    rev = rev * 10 + n % 10;                                                   \
    n /= 10;                                                                   \
  }                                                                            \
  n = rev
#define v_in2(x, y, n)                                                         \
  vll x(n), y(n);                                                              \
  _for(i, 0, n) cin >> x[i] >> y[i]
#define s_in(x, n)                                                             \
  sll x;                                                                       \
  for (i = 0, int a; i < n, cin >> a; i++)                                     \
  x.insert(a)
#define m_in(x, n)                                                             \
  mll x;                                                                       \
  for (i = 0, int a, b; i < n, cin >> a >> b; i++)                             \
  x[a] = b
#define v2_in(x, n, m)                                                         \
  vvi x(n, vi(m));                                                             \
  _for(i, 0, n) for (auto &j : x[i]) cin >> j
#define v2_out(x)                                                              \
  _for(i, 0, x.sz) {                                                           \
    for (auto j : x[i])                                                        \
      cout << j << " ";                                                        \
    cout << endl;                                                              \
  }
#define MAX 501
ll t, n, i, j, k, a, b, c, m, l, r, p, q, s;

vll x(MAX);
void solve() {
  _for(i, 3, 501) { x[i] = x[i - 1] + i * (i - 1); }
}
void dihan() {
  cin >> n;
  cout << x[n];
}

int main() {
  ____ solve();
  dihan();
}
