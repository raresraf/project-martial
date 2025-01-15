#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x = 0;
  while (n--) {
    string s;

    cin >> s;
    if (s[0] == '+')
      x++;
    if (s[0] == '-')
      x--;
    if (s[2] == '+')
      x++;
    if (s[2] == '-')
      x--;
  }
  cout << x;

  return 0;
}
