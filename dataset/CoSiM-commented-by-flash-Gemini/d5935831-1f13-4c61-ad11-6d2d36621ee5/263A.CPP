#include <iostream>

using namespace std;

int main() {
  int mat[5][5];

  for (auto i = 0; i < 5; i++) {
    for (auto j = 0; j < 5; j++) {
      cin >> mat[i][j];
    }
  }

  for (auto i = 0; i < 5; i++) {
    for (auto j = 0; j < 5; j++) {
      if (mat[i][j] == 1) {
        cout << abs(2 - i) + abs(2 - j) << endl;
      }
    }
  }

  return 0;
}
