#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int *a = new int[n];
  int i;
  for (i = 0; i < n; i++)
    cin >> a[i];
  int bar = a[k - 1], c = 0;
  for (i = 0; i < n; i++) {
    if (a[i] > 0 && a[i] >= bar)
      ++c;
  }
  cout << c << endl;
  delete a;
  a = NULL;

  return 0;
}