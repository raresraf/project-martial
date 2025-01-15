#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {

  int n, k;
  cin >> n >> k;

  vector<int> a(n);

  for (int i = 0; i < n; ++i)
    cin >> a[i];

  sort(a.begin(), a.end());
  reverse(a.begin(), a.end());

  int target = a[--k];
  int cnt = 0;

  for (int i = 0; i < n; ++i) {
    if (a[i] >= target && a[i] > 0) {
      cnt++;
    }
  }
  cout << cnt << endl;

  return 0;
}