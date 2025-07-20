#include <bits/stdc++.h>
#define pb push_back
#define lli long long int
using namespace std;

int main() {
  std::ios_base::sync_with_stdio(false);
  cin.tie(0);

  string s;
  cin >> s;
  string res = "";
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  for (int i = 0; i < s.length(); ++i) {
    if (!(s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
          s[i] == 'u' || s[i] == 'y')) {
      res += '.';
      res += s[i];
    }
  }
  cout << res << '\n';
  return 0;
}
