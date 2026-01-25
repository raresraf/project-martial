#include <iostream>
using namespace std;

int main() {
  int n;
  int x = 0, y = 0, z = 0;
  int vec[100][3];
  cin >> n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      cin >> vec[i][j];
    x += vec[i][0];
    y += vec[i][1];
    z += vec[i][2];
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
