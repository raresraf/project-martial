//
// Created by Taewoo Kim on 3/21/2017.
//

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int solvable = 0;
  for (int p = 0; p < n; p++) {
    int total = 0;
    for (int i = 0; i < 3; i++) {
      bool opinion;
      cin >> opinion;
      total += opinion;
    }
    if (total >= 2)
      ++solvable;
  }

  cout << solvable;

  return 0;
}