#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  int a[n + 2];
  string s;
  cin >> s;
  for (int i = 0; i < t; i++) {
    for (int i = 0; i < n; i++) {
      if (s[i] == 'B') {
        if (i + 1 < n && s[i + 1] == 'G') {
          swap(s[i], s[i + 1]);
          i++;
        }
      }
    }
  }
  cout << s << endl;
  return 0;
}