#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  string s;
  int n, a;

  cin >> n;

  if (n >= 1 && n <= 100) {
    for (int i = 0; i < n; i++) {
      cin >> s;
      if (s.length() > 10) {
        cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
      } else
        cout << s << endl;
    }
  }
}
