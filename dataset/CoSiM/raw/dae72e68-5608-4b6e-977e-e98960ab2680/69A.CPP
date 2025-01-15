#include <iostream>
using namespace std;
int main() {
  int force[101][3];
  int n;
  int x = 0, y = 0, z = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> force[i][0] >> force[i][1] >> force[i][2];
    x += force[i][0];
    y += force[i][1];
    z += force[i][2];
  }
  cout << ((x == 0 && y == 0 && z == 0) ? "YES" : "NO");
  return 0;
}
