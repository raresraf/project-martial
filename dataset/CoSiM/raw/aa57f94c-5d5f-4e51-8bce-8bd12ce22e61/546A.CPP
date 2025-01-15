/**
 * A. Nastya Is Reading a Book
 * https://codeforces.com/contest/1136/problem/A
 */

#include <iostream>
using namespace std;

int a[110][2];

int main() {
  int n, k;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a[i][0] >> a[i][1];
  }
  cin >> k;
  for (int i = 0; i < n; ++i) {
    if (k <= a[i][1]) {
      cout << n - i << endl;
      break;
    }
  }
  return 0;
}
