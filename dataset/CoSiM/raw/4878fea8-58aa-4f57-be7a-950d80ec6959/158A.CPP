#include <iostream>

using namespace std;

int main() {

  int n, k;
  int a[50];
  cin >> n >> k;
  int count = 0;
  int i;

  for (i = 0; i < n; i++)
    cin >> a[i];

  for (i = 0; i < n; i++) {

    if (a[i] >= a[k - 1] && a[i] > 0)
      count++;
  }

  cout << count;

  return 0;
}
