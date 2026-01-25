#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  int len = a.length();
  for (int i = 0; i < len; i++) {
    if ((int)tolower(a[i]) > (int)tolower(b[i])) {
      cout << 1;
      return 0;
    }
    if ((int)tolower(a[i]) < (int)tolower(b[i])) {
      cout << -1;
      return 0;
    }
  }

  cout << 0;
}
