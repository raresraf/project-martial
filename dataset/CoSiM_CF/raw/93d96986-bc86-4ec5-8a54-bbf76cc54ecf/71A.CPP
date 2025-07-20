#include <bits/stdc++.h>
using namespace std;
int main() {
  freopen("in", "r", stdin);
  char str[1000];
  int m, n, o, len;
  cin >> m;
  for (int i = 0; i < m; i++) {
    cin >> str;
    len = strlen(str);
    if (len > 10) {
      cout << str[0] << (len - 2) << str[len - 1] << endl;
    } else {
      cout << str << endl;
    }
  }
  return 0;
}
