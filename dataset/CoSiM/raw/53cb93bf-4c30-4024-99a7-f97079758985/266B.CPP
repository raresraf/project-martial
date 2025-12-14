//  [amitdu6ey]
#include <bits/stdc++.h>
#define hell 1000000009
#define bug(x) cout << "$ " << x << " $" << endl
#define ll long long
#define pb push_back
#define mp make_pair
#define tr(z) for (auto it = z.begin(); it != z.end(); it++)
#define loop(i, a, b) for (long long i = a; i < b; i++)
#define vbool vector<bool>
#define vvbool vector<vector<bool>>
#define vchar vector<char>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vll vector<long long>
#define vvl vector<vector<long long>>
#define ninf numeric_limits<int>::min()
#define pinf numeric_limits<int>::max()
using namespace std;

void solve() {
  ll n, t;
  string str;
  cin >> n >> t;
  cin >> str;
  vchar s(n);
  loop(i, 0, n) { s[i] = str[i]; }
  while (t--) {
    loop(i, 0, n - 1) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }
  loop(i, 0, n) { cout << s[i]; }
  return;
}

int main() {
  ios_base::sync_with_stdio(0);
  cout.tie(0);
  cin.tie(0);
  int test_cases = 1;
  // cin>>test_cases;
  while (test_cases--) {
    solve();
  }
  return 0;
}