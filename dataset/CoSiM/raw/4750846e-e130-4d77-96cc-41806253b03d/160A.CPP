#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, sum = 0;
  cin >> n;
  int i, a[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + n, greater<int>());
  int coins = 0, taken = 0;
  taken = a[0];
  coins = 1;
  i = 1;
  int j = n - 1;
  sum -= a[0];
  while (i <= j) {
    if (taken > sum)
      break;
    taken = taken + a[i];
    sum = sum - a[i];
    coins++;
    i++;
  }
  cout << coins;
}