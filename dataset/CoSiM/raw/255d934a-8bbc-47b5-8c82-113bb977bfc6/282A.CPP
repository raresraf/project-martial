#include <bits/stdc++.h>

using namespace std;

int main(void) {
  int t, x = 0;
  string o;
  cin >> t;

  while (t--) {
    cin >> o;
    if (o[1] == '+')
      x++;
    else if (o[1] == '-')
      x--;
  }

  cout << x;

  return 0;
}
