// 263A - Beautiful Matrix

#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int m[25][25], i, j;

  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      cin >> m[i][j];

  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      if (m[i][j] == 1)
        cout << abs(i - 2) + abs(j - 2);

  return 0;
}
