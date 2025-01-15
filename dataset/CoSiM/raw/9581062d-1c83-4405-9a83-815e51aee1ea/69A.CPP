#include <bits/stdc++.h>
using namespace std;
int main() {
  int x1, y1, z1, sum_1, sum_2, sum_3, n;
  while (cin >> n) {
    sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;
    while (n--) {
      cin >> x1 >> y1 >> z1;
      sum_1 += x1;
      sum_2 += y1;
      sum_3 += z1;
    }
    if (sum_1 == 0 && sum_2 == 0 && sum_3 == 0)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}
