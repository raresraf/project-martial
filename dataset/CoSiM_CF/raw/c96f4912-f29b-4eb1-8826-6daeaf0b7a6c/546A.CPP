#include <bits/stdc++.h>
using namespace std;
int main() {
  int x, y, z, sum = 0, i;
  cin >> x >> y >> z;
  for (i = 1; i <= z; i++) {
    sum += i * x;
  }
  if (sum <= y)
    cout << "0" << endl;
  else
    cout << sum - y << endl;
}
