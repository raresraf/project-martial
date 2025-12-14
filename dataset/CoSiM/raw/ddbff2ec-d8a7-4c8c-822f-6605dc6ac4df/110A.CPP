#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  string s;
  cin >> s;
  ll len = s.length();
  ll cnt = 0;
  for (int i = 0; i < len; i++) {
    if (s[i] == '7' || s[i] == '4') {
      cnt++;
    }
  }
  if (cnt == 4 || cnt == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
