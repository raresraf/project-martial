// 28.01.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int x[n];
  int y[n];
  int z[n];

  int sx = 0;
  int sy = 0;
  int sz = 0;

  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  for (int i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sz += z[i];
  }
  if (sx == 0 && sy == 0 && sz == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}