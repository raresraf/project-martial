#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int steps = 0;
  vector<vector<int>> array1(5, vector<int>(5, 0));
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> array1[i][j];
      if (array1[i][j] == 1) {
        steps = abs(3 - (j + 1)) + abs(3 - (i + 1));
      }
    }
  }
  cout << steps << endl;
  return 0;
}