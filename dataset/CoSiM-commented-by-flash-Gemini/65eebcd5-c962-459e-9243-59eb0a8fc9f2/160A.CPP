#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n = 0, a = 0, m = 0, sum = 0, k = 0, temp_sum = 0;
  cin >> n;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    cin >> a;
    v.push_back(a);
    m += a;
  }
  sort(v.begin(), v.end());
  reverse(v.begin(), v.end());
  for (int i = 0; i < n; i++) {
    temp_sum += v[i];
    k++;
    m = m - v[i];
    if (temp_sum > m)
      break;
  }
  cout << k;
  return 0;
}