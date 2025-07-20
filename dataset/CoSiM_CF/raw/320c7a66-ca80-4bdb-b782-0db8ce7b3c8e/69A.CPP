#include <iostream>

using namespace std;

int main() {
  int x = 0, y = 0, z = 0;
  int num;
  cin >> num;
  int coordinates[num][3];
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> coordinates[i][j];
    }
    x = x + coordinates[i][0];
    y = y + coordinates[i][1];
    z = z + coordinates[i][2];
  }
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
