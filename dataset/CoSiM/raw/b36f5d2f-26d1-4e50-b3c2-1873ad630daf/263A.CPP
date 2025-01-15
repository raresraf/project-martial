#include <bits/stdc++.h>
#define fastIO                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);

using namespace std;

int main() {
  fastIO int a, posi, posj, ans;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a;
      if (a == 1) {
        posi = i;
        posj = j;
      }
    }
  }
  ans = abs(posi - 3) + abs(posj - 3);
  cout << ans << endl;

  return 0;
}
