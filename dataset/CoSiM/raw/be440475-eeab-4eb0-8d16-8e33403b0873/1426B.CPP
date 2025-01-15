// Author - Soumik Roy
// Date & Time - October 02, 2020 12:14 PM
// Problem name - B. Symmetric Matrix
// Problem url - https://codeforces.com/contest/1426/problem/B
// Time limit - 1000 ms
// Memory Limit - 256 MB
// Parsed with ai-virtual-assistant
// https://github.com/Saurav-Paul/AI-virtual-assistant-python

#include <bits/stdc++.h>
#define ll long long int
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define MOD 1000000007
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define eb emplace_back
#define ull unsigned long long
#define bug(x) cout << "  [ " #x << " = " << x << " ]" << endl;
#define KAMEHAMEHA ios_base::sync_with_stdio(0);
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
int dir[] = {0, 1, 0, -1, 1, 0, -1, 0};
void solve() {
  int n, m;
  cin >> n >> m;
  bool checkeq = false;

  for (int i = 0; i < n; i++) {
    int a[4], x = 0;
    for (int j = 0; j < 4; j++) {
      cin >> a[j];
    }
    if (a[1] == a[2]) {
      checkeq = true;
    }
  }
  if (m % 2) {
    cout << "NO" << endl;
    return;
  }
  // cout<<m<<" "<<checkeq<<endl;
  if (!checkeq) {
    cout << "NO" << endl;
    return;
  }
  cout << "YES" << endl;
}
int main() {
  KAMEHAMEHA
  int t = 1;
  cin >> t;
  for (int cn = 1; cn <= t; cn++) {
    solve();
  }
  return 0;
}
