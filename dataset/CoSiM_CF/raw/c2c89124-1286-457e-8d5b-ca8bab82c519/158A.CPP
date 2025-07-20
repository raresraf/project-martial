#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n;
  cin >> k;

  int a[50];

  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  if (n == 1) {
    if (a[0] > 0)
      cout << 1;
    else
      cout << 0;
    return 0;
  }

  int num_passed = 0;
  for (int i = 0; i < n; i++) {
    if (a[i] == 0)
      break;

    if (a[i] >= a[k - 1])
      num_passed++;
  }

  cout << num_passed;
  return 0;
}
