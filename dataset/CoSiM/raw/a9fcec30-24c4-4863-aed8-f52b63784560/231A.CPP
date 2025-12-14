#include <iostream>
using namespace std;
int main() {
  int n, i, j, k, l = 0;
  cin >> n;
  int a[n][3];
  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++)
      cin >> a[i][j];
  for (i = 0; i < n; i++) {
    if ((a[i][0] == 1 && a[i][1] == 1) || (a[i][2] == 1 && a[i][1] == 1) ||
        (a[i][0] == 1 && a[i][2] == 1))
      l++;
  }

  cout << l;
  return 0;
}
