#include <iostream>
#include <vector>
using namespace std;

int main() {

  vector<vector<short>> matrix(5, vector<short>(5));
  for (short i = 0; i < 5; i++) {
    for (short j = 0; j < 5; j++) {
      cin >> matrix[i][j];
    }
  }

  for (short row{}; row < 5; row++) {
    for (short column{}; column < 5; column++) {

      if (matrix[row][column] == 1) {
        // to make 1 in the middle --> [row][column] = [2][2]
        short total_moves{};
        total_moves = abs(2.0 - row) + abs(2.0 - column);
        cout << total_moves << endl;
        return 0;
      }
    }
  }
}