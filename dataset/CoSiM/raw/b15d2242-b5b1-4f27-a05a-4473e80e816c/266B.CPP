#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t, i, k = 0;
  cin >> n >> t;
  char s[n];
  cin >> s;
  while (k < t) {
    i = 0;
    while (i < (n - 1)) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i = i + 2;
      } else {
        i++;
      }
    }
    k++;
  }
  cout << s;
  return 0;
}
