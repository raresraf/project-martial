/*
 *
 *    J1K7_7
 *
 */
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<long long> vll;
#define l(x) (x << 1) + 1
#define r(x) (x << 1) + 2
#define mid(l, r) ((l + r) >> 1)
#define mp make_pair
#define pb push_back
#define all(a) a.begin(), a.end()
#define debug(x)                                                               \
  { cerr << #x << " = " << x << "\n"; }
#define debug2(x, y)                                                           \
  { cerr << #x << " = " << x << ", " << #y << " = " << y << "\n"; }
#define ss second
#define ff first
#define m0(x) memset(x, 0, sizeof(x))
#define snuke(c, itr)                                                          \
  for (__typeof((c).begin()) itr = (c).begin(); itr != (c).end(); itr++)

template <class T> inline bool ispow2(T x) {
  return (x != 0 && (x & (x - 1)) == 0);
} // 0 or 1

template <class T> inline T powmod(T a, T b, T mod) {
  ll res = 1;
  while (b) {
    if (b & 1)
      res = (res * a) % mod;
    a = (a * a) % mod;
    b >>= 1;
  }
  return res;
}
template <class T> inline T gcd(T a, T b) {
  ll t;
  while (b) {
    a = a % b;
    t = a;
    a = b;
    b = t;
  }
  return a;
}
template <class T> inline T lcm(T a, T b) { return a / gcd(a, b) * b; }

inline int nextint() {
  int x;
  scanf("%d", &x);
  return x;
}
inline ll nextll() {
  ll x;
  scanf("%lld", &x);
  return x;
}

const int mod = 1e9 + 7;

const ll mx_ll = numeric_limits<ll>::max();
const int mx_int = numeric_limits<int>::max();

const long double PI = (long double)(3.1415926535897932384626433832795);

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  ll l, r, k;
  cin >> l >> r >> k;
  vector<ll> ans;
  ll curr = 1ll;
  ll val = k;

  while (1) {
    if (curr >= l && curr <= r) {
      ans.pb(curr);
    }
    if (curr > mx_ll / k)
      break;
    curr *= k;
  }
  if (ans.size() == 0) {
    cout << "-1";
    return 0;
  }
  for (auto x : ans)
    cout << x << " ";

  return 0;
}
