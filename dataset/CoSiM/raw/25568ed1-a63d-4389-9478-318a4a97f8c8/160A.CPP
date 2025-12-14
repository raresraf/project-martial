// Greedy - Sorting.

#include <algorithm>
#include <iostream>

using namespace std;

int main(void) {
  int n, totalSum = 0, count = 0, countSum = 0;
  cin >> n;
  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    totalSum += a[i];
  }
  sort(a, a + n, greater<int>());
  for (int i = 0; i < n; i++) {
    if (countSum <= totalSum / 2) {
      countSum += a[i];
      count++;
    } else {
      break;
    }
  }
  cout << count << endl;

  return 0;
}