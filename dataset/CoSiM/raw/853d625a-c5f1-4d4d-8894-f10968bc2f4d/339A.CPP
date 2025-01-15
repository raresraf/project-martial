#include <bits/stdc++.h>
using namespace std;
int main() {
  int x = 0, y = 0, z = 0;
  string sum;
  cin >> sum;
  for (int i = 0; i < sum.length(); i++) {
    if (sum[i] == '1')
      x = x + 1;
    else if (sum[i] == '2')
      y = y + 1;
    else if (sum[i] == '3')
      z = z + 1;
  }

  int b[3] = {x, y, z};
  for (int j = 0; j < x; j++) {
    if (x + y + z == j + 1)
      cout << "1";
    else
      cout << "1+";
  }

  for (int f = 0; f < y; f++) {
    if (x + y + z == f + b[0] + 1)
      cout << "2";
    else
      cout << "2+";
  }

  for (int c = 0; c < z; c++) {
    if (x + y + z == c + b[0] + b[1] + 1)
      cout << "3";
    else
      cout << "3+";
  }
  return 0;
}
