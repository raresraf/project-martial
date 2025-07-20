//
// Created by Conor on 5/27/2020.
//

#include <iostream>
using namespace std;

int main() {
  int n;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> n;
      if (n) {
        cout << abs(2 - i) + abs(2 - j);
        return 0;
      }
    }
  }
}