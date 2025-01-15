#include <iostream>
using namespace std;
int main() {
  int a[100] = {0};
  int c, k, j = 0;
  cin >> c >> k;
  for (int i = 0; i < c; i++)
    cin >> a[i];
  for (int i = 0; i < c; i++) {
    if (a[i] >= a[k - 1] && a[i] != 0)
      j++;
  }
  cout << j;
  return 0;
}
