#include <bits/stdc++.h>
using namespace std;

#define pb push_back
#define ff first
#define ss second
#define ll long long
#define f(i, n) for (int i = 0; i < n; i++)
#define F(i, n) for (ll i = 0; i < n; i++)
#define fx(i, x, n) for (int i = x; i < n; i++)
#define Fx(i, x, n) for (ll i = x; i < n; i++)
#define endl '\n'
#define mod 1000000007
#define rex                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0)
#define all(x) x.begin(), x.end()
#define sortall(x) sort(all(x))
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef vector<string> vs;
typedef vector<pii> vpii;
typedef vector<pll> vpll;
const double PI = 3.141592653589793238460;

int main() {
  rex;

  int t;
  cin >> t;
  vi x(t), y(t), z(t);
  int sumx = 0, sumy = 0, sumz = 0;
  f(i, t) cin >> x[i] >> y[i] >> y[i];
  sumx = accumulate(all(x), 0);
  sumy = accumulate(all(y), 0);
  sumz = accumulate(all(z), 0);
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}