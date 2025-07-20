#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int x[n];
  int y[n];
  int z[n];

  int sumx = 0;
  int sumy = 0;
  int sumz = 0;

  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  if (sumz == 0 && sumy == 0 && sumx == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
