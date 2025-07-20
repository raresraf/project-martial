#include <bits/stdc++.h>
using namespace std;
#define ll long long
int main() {
  string s;
  cin >> s;
  ll len = (ll)s.size();
  ll cnt = 0;
  for (ll i = 0; i < len; i++) {
    if (s[i] == '4' || s[i] == '7') {
      cnt++;
    }
  }
  if (cnt == 4 || cnt == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}