/*
        Author   : Abhinav

        Modified :

*/

#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<ll> vi;
typedef map<ll, ll> mllmp;
typedef pair<int, int> PII;

#define faster ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0);
#define PB push_back
#define MP make_pair
#define INF 0x3f3f3f3f

const ll mod = 1e9 + 7;

#define rep(i, j, k) for (ll i = j; i < k; i++)
#define repv(i, j, k) for (ll i = j; i > k; i--)

int main() {
  faster;
  ll n, x = 0;
  cin >> n;
  string s;
  while (n--) {
    cin >> s;
    if (s[0] == '+' || s[2] == '+')
      x++;
    else
      x--;
  }
  cout << x;
  return 0;
}