#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;

  string s;
  cin >> s;

  if (n == 1) {
    cout << s << endl;
  } else {
    while (t--) {
      for (int i = 0; i < n;) {
        if (s[i] == 'B' && s[i + 1] == 'G') {
          swap(s[i], s[i + 1]);
          i += 2;
        } else {
          i++;
        }
      }
    }
    cout << s << endl;
  }
}
