#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string s;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s;
    int l = s.size();
    if (l > 10) {
      cout << s[0] << l - 2 << s[l - 1] << "\n";
    } else {
      cout << s << "\n";
    }
  }
}
