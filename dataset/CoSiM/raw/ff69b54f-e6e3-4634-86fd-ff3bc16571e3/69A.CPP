#include <iostream>
using namespace std;

int main() {
  int n, i, j, s = 0, a[100][3], f = 1;
  cin >> n;
  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++)
      cin >> a[i][j];
  for (j = 0; j < 3; j++) {
    for (i = 0; i < n; i++)
      s += a[i][j];
    if (s) {
      f = 0;
      break;
    }
  }
  if (f)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
