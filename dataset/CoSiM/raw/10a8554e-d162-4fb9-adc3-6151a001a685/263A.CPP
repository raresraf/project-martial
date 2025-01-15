#include <iostream>
using namespace std;
int mod(int k, int l) {
  if ((k - l) >= 0)
    return k - l;
  else
    return l - k;
}
int main() {
  int a[5][5], i, j, sum, x, y;
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      cin >> a[i][j];
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++)
      if (a[i][j] == 1)
        x = i, y = j;

  ++x, ++y;
  sum = mod(3, x) + mod(3, y);
  cout << sum;
}
