#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x = 0;
  while (n--) {
    string s;
    cin >> s;
    int p = -1;
    for (int i = 0; i < 3; i++) {
      if (s[i] == '+') {
        p = 0;
        break;
      }
      if (s[i] == '-') {
        p = 1;
        break;
      }
    }
    if (p == 0)
      x++;
    else
      x--;
  }
  cout << x;
  return 0;
}