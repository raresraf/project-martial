#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, j, a;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a;
      if (a == 1) {
        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }
  return 0;
}
