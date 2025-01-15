#include <bits/stdc++.h>
using namespace std;

int main() {
  int temp;
  for (int i = 1; i < 26; ++i) {
    cin >> temp;
    if (temp == 1) {
      int rows;
      if ((i % 5) == 0)
        rows = i / 5;
      else
        rows = i / 5 + 1;
      int col = i - (rows - 1) * 5;

      cout << abs(rows - 3) + abs(col - 3) << "\n";
      return 0;
    }
  }
  return 0;
}