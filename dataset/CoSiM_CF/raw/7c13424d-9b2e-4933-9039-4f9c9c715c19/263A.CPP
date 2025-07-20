#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  for (int row = 0; row < 5; row++) {
    for (int col = 0; col < 5; col++) {
      cin >> n;
      if (n == 1) {
        cout << abs(2 - row) + abs(2 - col);
      }
    }
  }

  return 0;
}
