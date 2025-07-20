#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int x, y;
  int t;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> t;
      if (t) {
        x = j + 1;
        y = i + 1;
      }
    }
  }
  cout << abs(3 - x) + abs(3 - y) << "\n";
  return 0;
}
