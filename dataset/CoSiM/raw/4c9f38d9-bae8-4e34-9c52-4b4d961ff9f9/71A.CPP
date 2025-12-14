#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int l = s.size();
    if (l <= 10) {
      cout << s << endl;
    } else {
      cout << s.substr(0, 1) << l - 2 << s.substr(l - 1, l - 1) << endl;
      ;
    }
  }

  return 0;
}
