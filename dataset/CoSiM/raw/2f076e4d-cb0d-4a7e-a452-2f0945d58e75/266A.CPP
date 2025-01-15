//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  string stones;
  int n;
  cin >> n;
  cin >> stones;

  int count;
  count = 0;

  for (int i = 1; i < stones.size(); ++i) {
    if (stones[i] == stones[i - 1]) {
      count++;
    }
  }

  cout << count;

  return 0;
}