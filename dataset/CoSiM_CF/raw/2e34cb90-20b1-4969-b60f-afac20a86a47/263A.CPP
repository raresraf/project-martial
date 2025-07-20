#include <iostream>
using namespace std;
int main() {
  int matrix[25][25], count;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      if (matrix[i][j] == 1)
        count = abs(j - 2) + abs(i - 2);
    }
  }
  cout << count;
  return 0;
}
