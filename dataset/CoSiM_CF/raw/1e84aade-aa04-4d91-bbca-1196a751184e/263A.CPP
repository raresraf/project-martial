//
// Created by Taewoo Kim on 3/24/2017.
//

#include <iostream>

using namespace std;

int main() {
  int pos = 0;
  for (int i = 0; cin >> i; pos++) {
    if (i == 1) {
      int r = pos % 5 + 1;
      int c = pos / 5 + 1;
      cout << abs(r - 3) + abs(c - 3);
      break;
    }
  }
  return 0;
}