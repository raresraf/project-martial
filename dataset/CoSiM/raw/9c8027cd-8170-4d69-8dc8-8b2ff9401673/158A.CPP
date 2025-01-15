// Implementation.

#include <iostream>

using namespace std;

int main(void) {
  int n, k, a[101];
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  int count = 0, pointsAtK = a[k - 1];

  for (int i = 0; i < n; i++) {
    if (a[i] >= pointsAtK && a[i] != 0)
      count++;
    else
      break;
  }
  cout << count << endl;

  return 0;
}