#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, *a, i, j, t1 = 0, t2 = 0;
  cin >> n;
  a = new int[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
    t2 += a[i];
  }
  sort(a, a + n, greater<int>());
  for (i = 0; i < n; i++) {
    t1 += a[i];
    if (t1 > (t2 - t1))
      break;
  }
  cout << i + 1 << endl;
}