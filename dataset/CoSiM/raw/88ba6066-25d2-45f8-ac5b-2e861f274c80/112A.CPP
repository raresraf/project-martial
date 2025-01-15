/*
        Author   : Abhinav

        Modified : 30-08-2018 08:09:43 PM

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
  string s1, s2;
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  if (s1 == s2)
    cout << "0";
  else if (s1 < s2)
    cout << "-1";
  else
    cout << "1";
  return 0;
}