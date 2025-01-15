#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, k, ara[50];
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> ara[i];
  }
  if (ara[k - 1] > 0) {
    while (k < n && ara[k - 1] == ara[k])
      k++;
  } else if (ara[k - 1] < 1) {
    while (k >= 1 && ara[k - 1] < 1)
      k--;
  }
  if (k > 0)
    cout << k;
  else
    cout << 0;
  return 0;
}
