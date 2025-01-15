#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int n, t;
  string s, res;
  cin >> n >> t >> s;

  while (t--) {
    for (int i = 0; i < n; ++i) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        char temp;
        temp = s[i + 1];
        s[i + 1] = s[i];
        s[i] = temp;
        i += 1;
      }
    }
  }
  cout << s << '\n';
  return 0;
}
