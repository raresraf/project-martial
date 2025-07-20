#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, greatsum, cumsum;
  int a[100];
  cin >> n;
  int i, j, k;
  cumsum = 0;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    cumsum += a[i];
  }
  int count;
  count = 0;
  greatsum = 0;
  sort(a, a + n);
  for (i = n - 1; i >= 0; i--) {
    if (greatsum <= cumsum) {
      greatsum = greatsum + a[i];
      cumsum = cumsum - a[i];
      count++;
    } else {
      break;
    }
  }
  cout << count << endl;
  return 0;
}
