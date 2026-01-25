#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x, y, z, sum = 0, count = 0;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sum = x + y + z;
    if (sum >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}