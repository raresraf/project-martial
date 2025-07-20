#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      scanf("%d", &n);
      if (n) {
        cout << abs(3 - i) + abs(3 - j);
        break;
      }
    }
  }
}
