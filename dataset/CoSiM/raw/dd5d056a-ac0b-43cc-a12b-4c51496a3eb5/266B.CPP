#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  vector<char> v;
  ll n, t;
  cin >> n >> t;
  vector<char> s;
  ll len = (ll)s.size();
  s.push_back('1');
  for (ll k = 0; k < n; k++) {
    char c;
    cin >> c;
    s.push_back(c);
  }
  for (ll i = 0; i < t; i++) {
    for (ll j = 1; j <= n - 1;) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        swap(s[j], s[j + 1]);
        j++;
      }
      j++;
    }
  }
  for (auto x : s) {
    if (x != '1')
      cout << x;
  }
  cout << endl;
  return 0;
}