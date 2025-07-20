/*-----rk_98-----*/
#include <bits/stdc++.h>
#define fast()                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define ll long long
#define mod 1000000007
#define eps 1e-9
#define ff first
#define ss second
#define pb push_back
#define endl "\n"
using namespace std;
int main() {
  // fast();
  ll n;
  cin >> n;
  ll sum = 0, arr[n];
  for (ll i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n, greater<ll>());
  ll twin = 0, coin = 0;
  for (ll i = 0; i < n; i++) {
    twin += arr[i];
    sum -= arr[i];
    coin++;
    if (twin > sum)
      break;
  }
  cout << coin << endl;
}
