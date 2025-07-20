#include <bits/stdc++.h>
using namespace std;

int main() {

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      int t;
      cin >> t;

      if (t != 0) {
        int p = abs(i - 2) + abs(j - 2);
        cout << p << endl;
        return 0;
      }
    }
  }

  return 0;
}
