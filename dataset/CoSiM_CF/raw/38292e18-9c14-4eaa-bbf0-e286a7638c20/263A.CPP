#include <iostream>

using namespace std;

int main() {
  int m[5][5], l, c, dist;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> m[i][j];
      if (m[i][j] == 1) {
        l = i;
        c = j;
      }
    }
  }

  dist = abs(2 - l) + abs(2 - c);

  cout << dist << endl;

  return 0;
}