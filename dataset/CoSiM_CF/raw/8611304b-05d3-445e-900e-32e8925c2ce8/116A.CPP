#include <iostream>

using namespace std;

int main() {
  int n, a[20000], b[20000], i, sum = 0, max = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum -= a[i];
    cin >> b[i];
    sum += b[i];
    if (max < sum) {
      max = sum;
    }
  }
  cout << max;
}
