#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, j, k, n, t;
  while (cin >> n >> t) {
    string s;
    cin >> s;
    for (i = 0; i < t; i++) {
      for (j = 0; j < n - 1; j++) {
        if (s[j] == 'B' && s[j + 1] == 'G') {
          s[j] = 'G';
          s[j + 1] = 'B';
          j++;
        }
      }
    }
    cout << s << endl;
  }
  return 0;
}
