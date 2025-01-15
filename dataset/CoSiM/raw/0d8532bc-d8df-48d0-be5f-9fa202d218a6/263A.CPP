#include <bits/stdc++.h>

using namespace std;

int main() {

  int x, y;
  int ans;

  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      cin >> x;
      if (x == 1) {
        ans = abs(2 - i) + abs(2 - j);
      }
    }

  cout << ans << endl;

  return 0;
}
