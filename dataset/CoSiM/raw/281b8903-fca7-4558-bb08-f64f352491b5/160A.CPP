#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
typedef long double ld;
typedef pair<ll, ll> pll;
#define pb push_back
#define mp make_pair
#define f first
#define s second
#define gcd(a, b) __gcd(a, b)
#define lpr(i, s, e) for (ll i = s; i >= e; i--)
#define lpi(i, s, e) for (ll i = s; i <= e; i++)
#define lp(n) for (ll i = 0; i < n; i++)
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  int n, ans = 0, sum = 0, me = 0;
  cin >> n;
  int arr[n];
  lp(n) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n, greater<int>());
  lp(n) {
    if (me > sum)
      break;
    ans++;
    me += arr[i];
    sum -= arr[i];
  }
  cout << ans << endl;
  return 0;
}