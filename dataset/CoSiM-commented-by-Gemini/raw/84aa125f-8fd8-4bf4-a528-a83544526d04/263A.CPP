#include <iostream>

int main() {
  int arr[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cin >> arr[i][j];
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (arr[i][j] == 1) {
        int diff_x = abs(i - 2);
        int diff_y = abs(j - 2);
        std::cout << diff_x + diff_y;
      }
    }
  }
}