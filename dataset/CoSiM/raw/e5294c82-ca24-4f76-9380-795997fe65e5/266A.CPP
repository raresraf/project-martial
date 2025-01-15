#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, cnt = 0;
  cin >> n;
  string s;
  cin >> s;
  for (int i = 0; i < n; i++) {
    if ((s[i] == 'R' && s[i + 1] == 'R') || (s[i] == 'G' && s[i + 1] == 'G') ||
        (s[i] == 'B' && s[i + 1] == 'B')) {
      cnt++;
    } else {
      continue;
    }
  }
  cout << cnt << endl;
}
