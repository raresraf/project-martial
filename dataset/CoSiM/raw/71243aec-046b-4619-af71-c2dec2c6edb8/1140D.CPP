#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
const ll mod = 1000000007;
typedef long double ld;
typedef pair<ll, ll> pll;
#define pb push_back
#define mp make_pair
#define f first
#define s second
#define gcd(a, b) __gcd(a, b)
#define lp(n) for (ll i = 0; i < n; i++)
ll solve(ll arr[], ll n, ll i, ll j) {
  if (j < i + 2)
    return 0;
  ll ans = 1e18;
  for (ll k = i + 1; k < j; k++)
    ans = min(ans, solve(arr, n, i, k) + solve(arr, n, k, j) + i * j * k);
  return ans;
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  ll n;
  cin >> n;
  ll arr[n + 1];
  for (ll i = 1; i <= n; i++)
    arr[i] = 1;
  // cout<<solve(arr,n,1,n)<<endl;
  ll ans = 0, x = 1, y = 2, z = 3;
  for (ll i = 1; i <= n - 2; i++) {
    ans += z * y * x;
    y++;
    z++;
  }
  cout << ans << endl;
  return 0;
}
