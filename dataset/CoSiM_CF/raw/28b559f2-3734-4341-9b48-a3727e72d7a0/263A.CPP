#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[60][60], ans = 0;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1)
        ans = abs(i - 3) + abs(j - 3);
    }
  }

  cout << ans << endl;

  // your code goes here
  return 0;
}
