#include <bits/stdc++.h>

using namespace std;

int main() {
  int num, x, y, z, sum_x = 0, sum_y = 0, sum_z = 0;
  cin >> num;

  for (int i = 0; i < num; i++) {
    scanf("%d %d %d", &x, &y, &z);
    sum_x += x;
    sum_y += y;
    sum_z += z;
  }
  if (sum_x == 0 && sum_y == 0 && sum_z == 0)
    cout << "YES";
  else
    cout << "NO";
}
