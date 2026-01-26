#include <iostream>
#include <math.h>
#include <vector>
using namespace std;
int main() {
  vector<vector<int>> input(5, vector<int>(5, 0));
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> input[i][j];
      if (input[i][j] == 1) {
        cout << abs(2 - i) + abs(2 - j);
        break;
      }
    }
  }
  return 0;
}