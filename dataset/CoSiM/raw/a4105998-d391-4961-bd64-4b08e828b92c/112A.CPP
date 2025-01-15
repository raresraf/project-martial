#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<ll, ll> pii;
typedef pair<double, ll> pdl;
typedef vector<ll> vll;
#define ff first
#define ss second
#define pb(x) push_back()
#define sl(x) scanf("%lld", &x)
#define sll(x, y) scanf("%lld%lld", &x, &y)
#define slll(x, y, z) scanf("%lld%lld%lld", &x, &y, &z)
#define pl(x) printf("%lld\n", x)
#define mp make_pair
ll mod = (ll)1e9 + 7;
const ll INF = (ll)1e18;
#define all(x) x.begin(), x.end()
#define s(x) ((ll)x.size())
#define tc                                                                     \
  ll t;                                                                        \
  cin >> t;                                                                    \
  while (t--)
#define FOR(i, j, n) for (ll i = j; i < n; i++)
#define ig cin.ignore()
#define fast_cin ios_base::sync_with_stdio(false)
ll power(ll a, ll b) {
  ll ret = 1;
  while (b) {
    if (b & 1)
      ret *= a;
    a *= a;
    if (ret >= mod)
      ret %= mod;
    if (a >= mod)
      a %= mod;
    b >>= 1;
  }
  return ret;
}
ll inv(ll x) { return power(x, mod - 2); }
#define run(m) for (auto it = m.begin(); it != m.end(); it++)
/*---------------------------------------------------------//
const int MAXM = 1e7+3010;   ll fact[MAXM];
void factorial(){ fact[0]=1;FOR(i,1,MAXM) fact[i]=(fact[i-1]*i)%mod;}
ll nCr(ll n,ll r){return
((fact[n])*(((inv(fact[r]))*(inv(fact[n-r])))%mod))%mod;}
// freopen("Task.in","r",stdin);freopen("Task.out","w",stdout);
//-------------------------------------------------------------------/*/

int main() {
  fast_cin;
  string s1;
  cin >> s1;
  string s2;
  cin >> s2;

  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  if (s1 == s2) {
    cout << "0\n";
    return 0;
  }

  FOR(i, 0, min(s1.size(), s2.size())) {
    if (s1[i] == s2[i])
      continue;

    if (s1[i] < s2[i])
      cout << "-1\n";
    else if (s1[i] > s2[i])
      cout << "1\n";
    return 0;
  }

  if (s1.size() > s2.size())
    cout << "1\n";
  else
    cout << "-1\n";
}
