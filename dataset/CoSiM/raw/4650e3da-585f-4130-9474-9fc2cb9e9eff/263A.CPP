#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int row = 0;
  int col = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      int input;
      cin >> input;
      if (input == 1) {
        row = i;
        col = j;
      }
    }
  }
  cout << abs(row - 2) + abs(col - 2) << endl;

  return 0;
}
