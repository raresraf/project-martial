#include <iostream>

using namespace std;

int main() {
  int b{0}, row{0}, col{0};
  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      cin >> b;
      if (b == 1) {
        row = i;
        col = j;
      }
    }
  }
  int pos = abs(row - 3) + abs(col - 3);
  cout << pos;
  return 0;
}
