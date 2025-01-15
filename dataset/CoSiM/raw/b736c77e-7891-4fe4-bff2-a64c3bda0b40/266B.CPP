#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t, i, j, k;
  cin >> n >> t;
  string s, a;
  char temp;
  cin >> s;
  for (i = 1; i <= t; i++) {
    for (j = s.size(); j > 0; j--) {
      if (s[j] == 'G' && s[j - 1] == 'B') {
        temp = s[j];
        s[j] = s[j - 1];
        s[j - 1] = temp;
        j = j - 1;
      }
    }
  }
  cout << s << endl;

  return 0;
}
