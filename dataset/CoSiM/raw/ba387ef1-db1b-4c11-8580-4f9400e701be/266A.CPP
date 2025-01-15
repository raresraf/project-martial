#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int a, i, j, arr[100], sum = 0, n = 0, dum = 0, v;
int main() {
  cin >> a;
  for (i = 0; i < a; i++) {
    cin >> arr[i];
    dum = dum + arr[i];
  }
  // dum >>= 1;
  v = dum / 2;
  sort(arr, arr + a);
  for (i = a - 1; i >= 0; i--) {
    sum = sum + arr[i];
    { n++; }
    if (sum > v) {
      break;
    }
  }
  cout << n << endl;
  return 0;
}
