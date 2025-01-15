#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define pb push_back
#define mod 1000000007
#define inf 1e8
#define pi pair<ll, ll>
#define pii pair<pi, ll>
#define f first
#define s second
#define mp make_pair
#define rep(i, n) for (ll i = 0; i < n; i++)
int main() {
  int n, t;
  string s;
  cin >> n >> t;
  cin >> s;
  int i = 0;
  while (t--) {
    i = 0;
    while (i <= n - 2) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i = i + 2;
      } else
        i++;
    }
  }
  cout << s;
}
