// please refer from line no. 207
/////*31022618*/////
//_MONU KUMAR
#include <bits/stdc++.h>
using namespace std;
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define tc                                                                     \
  ll t = 1;                                                                    \
  cin >> t;                                                                    \
  while (t--)
#define ll long long int
#define ld long double
#define ull unsigned long long
#define pll pair<ll, ll>
#define vi std::vector<ll>
#define vp vector<pll>
#define mll std::map<ll, ll>
#define mpl                                                                    \
  std:                                                                         \
  map<pll, ll>
#define p1(x) cout << x << "\n";
#define p2(x, y) cout << x << " " << y << "\n";
#define p3(x, y, z) cout << x << " " << y << " " << z << "\n";
#define p4(x, y, z, w) cout << x << " " << y << " " << z << " " << w << "\n";
#define trace(x)                                                               \
  cout << #x << ": " << x << " "                                               \
       << "\n";
#define trace2(x, y)                                                           \
  cout << #x << ": " << x << " | " << #y << ": " << y << "\n";
#define trace3(x, y, z)                                                        \
  cout << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << "\n";
#define trace4(a, b, c, d)                                                     \
  cout << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": "   \
       << c << " | " << #d << ": " << d << "\n";
#define nL cout << "\n";
#define lp(i, l, r) for (long long int i = l; i < r; i++)
#define lpr(i, r, l) for (long long int i = r - 1; i >= l; i--)
#define mem(a, x) memset(a, x, sizeof(a)) // set elements of array to some value
#define pi 3.1415926535897932384626
#define mod 1000000007
#define big 1e18
#define small -big
#define mp(a, b) make_pair(a, b)
#define pb(x) push_back(x)
#define ff first
#define ss second
#define siz 256
#define leadzero(a) __builtin_clz(a) // count leading zeroes
#define trailzero(a) __builtin_ctz(a) // count trailing zeroes
#define bitcount(a) __builtin_popcount(a) // count set bits
#define all(v) v.begin(), v.end()
#define lps(v, x) (lower_bound(all(v), x) - v.begin())
#define ups(v, x) (upper_bound(all(v), x) - v.begin())
#define ina(a, n)                                                              \
  ll a[n];                                                                     \
  lp(i, 0, n) cin >> a[i];
#define ona(a, n) lp(i, 0, n) cout << a[i] << " ";
//#########################################################################################
bool anagram(char *s1, char *s2) {
  ll count[siz];
  mem(count, 0);
  ll i;
  for (i = 0; s1[i] && s2[i]; i++) {
    count[s1[i]]++;
    count[s2[i]]--;
  }
  if (s1[i] || s2[i]) {
    return false;
  }
  lp(i, 0, siz) {
    if (count[i]) {
      return false;
    }
  }
  return true;
}
//###########################################################################################
// modulo Multiplication
ll moduloMultiplication(ll a, ll b, ll zz) {
  ll res = 0;
  a %= mod;
  while (b) {
    if (b & 1)
      res = (res + a) % zz;

    a = (2 * a) % zz;

    b >>= 1;
  }
  return res;
}
//###########################################################################################

ll convert(string s) {
  bool o = true;
  ll ans = 0;
  for (ll i = 0; i < s.length(); i++) {
    ans = ans * 10 + (s[i] - 48);
  }
  return ans;
}
//###########################################################################################
// function to calculate a^b under mod...
ll powmod(ll a, ll b, ll modulo) {
  if (b == 0 || a == 1)
    return 1;
  ll half = powmod(a, (b / 2), modulo) % modulo;
  ll full = (half * half) % modulo;

  if (b % 2)
    return (full * a) % modulo;
  return full % modulo;
}
//###########################################################################################
// function to calculate inverse modulo under mod...assuming gcd is 1
ll invmod(ll a, ll modulo) {
  ll check = powmod(a, modulo - 2, modulo) % modulo;
  return check;
}
//###########################################################################################

ll max(ll a, ll b) {
  if (a > b) {
    /* code */
    return a;
  }
  return b;
}
//###########################################################################################
ll min(ll a, ll b) {
  if (a < b) {
    /* code */
    return a;
  }
  return b;
}
//###########################################################################################
ll min(ll a, ll b, ll c) {
  if (a < b) {
    /* code */
    return min(a, c);
  } else {
    return min(b, c);
  }
}
//###########################################################################################
ll max(ll a, ll b, ll c) {
  if (a < b) {
    /* code */
    return max(b, c);
  } else {
    return min(a, c);
  }
}
//###########################################################################################
ll min(ll a, int b) {
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
ll min(int a, ll b) {
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
ll max(ll a, int b) {
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
ll max(int a, ll b) {
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
ll gcd(ll a, ll b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}
//###########################################################################################
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
//###########################################################################################
void yes() {
  cout << "YES"
       << "\n";
}
//###########################################################################################
void no() {
  cout << "NO"
       << "\n";
}
//###########################################################################################
int main() {
  fast
      //  clock_t launch=clock();
      //  string r = s1.substr(1, 3);
      //  while(clock()<CLOCKS_PER_SEC*10)
      //  using num = modnum<int(1e9)+7>;
      //  num len = s.size();
      //  double PI= 2 * acos(0.0); or M_PI
      ll n,
      k, ans = 0, flag = 0, temp = 0, f1 = 0, count = 0, f2 = 0, sum = 0,
         xx = 0, minn1 = big, minn2 = big, maxx1 = small, maxx2 = small;
  std::string s, ss;
  cin >> s >> ss;
  transform(all(s), s.begin(), ::tolower);
  transform(all(ss), ss.begin(), ::tolower);
  if (s < ss) {
    /* code */
    p1(-1)
  } else if (s == ss) {
    p1(0)
  } else {
    p1(1)
  }

  // clog<<((long double)(clock()-launch)/CLOCKS_PER_SEC)<<"\n";
  return 0;
}
//////////*********************end of program*********************//////////
