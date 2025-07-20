#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> a(n, 0);
  vector<int> b(n, 0);
  int count = 0, max = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
    if (a[i] > 0) {
      count -= a[i];
    }
    if (b[i] > 0) {
      count += b[i];
    }
    if (count > max) {
      max = count;
    }
  }
  cout << max << endl;

  return 0;
}