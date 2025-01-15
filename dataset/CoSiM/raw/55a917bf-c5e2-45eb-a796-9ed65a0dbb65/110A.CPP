#include <bits/stdc++.h>
#define endl "\n"
#define lli long long int
#define ulli unsigned long long int

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  string s;
  cin >> s;

  lli count = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      count++;
    }
  }

  s = to_string(count);

  bool luckynumber = true;
  for (int i = 0; i < s.size(); i++) {
    if (!(s[i] == '4' || s[i] == '7')) {
      luckynumber = false;
      break;
    }
  }

  if (luckynumber)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}