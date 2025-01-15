#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <vector>
using namespace std;
int main() {
  int n, k, c;
  cin >> n >> k >> c;
  int array[c];
  for (int i = 0; i < c; ++i)
    cin >> array[i];
  int cnt = 0;
  for (int i = 0; i <= n; i += k) {
    if (i != 0)
      cnt++;

    for (int j = 0; j < c; ++j) {
      if (array[j] < i + k && array[j] >= i + 1) {
        i = array[j];
        i -= k;
        break;
      }
    }
  }
  cout << cnt;
  return 0;
}