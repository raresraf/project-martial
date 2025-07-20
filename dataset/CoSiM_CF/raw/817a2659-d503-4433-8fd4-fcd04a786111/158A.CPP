#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  int n, k, key, ele;
  cin >> n >> k;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    cin >> ele;
    if (i + 1 == k)
      key = ele;
    if (ele <= 0)
      continue;
    v.push_back(ele);
  }
  if (key <= 0) {
    cout << v.size() << endl;
  } else {
    vector<int>::iterator uitr =
        lower_bound(v.begin(), v.end(), key - 1, greater<int>());
    cout << uitr - v.begin() << endl;
  }
  return 0;
}
