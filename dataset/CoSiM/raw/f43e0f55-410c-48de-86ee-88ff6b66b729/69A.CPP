// 69A - Young Physicist

#include <iostream>

using namespace std;

int main() {
  int n, i, sumx = 0, sumy = 0, sumz = 0;
  int x[100] = {0}, y[100] = {0}, z[100] = {0};

  cin >> n;

  for (i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];

  for (i = 0; i < n; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
