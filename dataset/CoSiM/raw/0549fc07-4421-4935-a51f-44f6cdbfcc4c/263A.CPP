#include <iostream>

using namespace std;

int main() {
  int m[5][5] = {0};
  int idx_i, idx_j;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> m[i][j];
      if (m[i][j] == 1) {
        idx_i = i;
        idx_j = j;
      }
    }
  }
  cout << abs(idx_i - 2) + abs(idx_j - 2) << endl;
  return 0;
}