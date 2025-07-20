#include <iostream>

using namespace std;

int main() {
  int idx, idy, temp;
  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      cin >> temp;
      if (temp == 1) {
        idx = i;
        idy = j;
      }
    }
  }

  cout << (abs(3 - idx) + abs(3 - idy)) << endl;
}