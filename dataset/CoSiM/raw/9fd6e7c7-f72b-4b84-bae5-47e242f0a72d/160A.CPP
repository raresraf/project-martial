#include <bits/stdc++.h>
#define LOG(x) cout << x << "\n"

using namespace std;

int main() {
  int n, a;
  vector<int> arr;
  int sum1;
  int sum2 = 0;
  cin >> n;

  int count = n;
  int size = n;

  while (n--) {
    cin >> a;
    arr.push_back(a);
  }

  sum1 = accumulate(arr.begin(), arr.end(), 0);

  while (count--) {
    if (sum1 < sum2) {
      break;
    }
    auto itMax = max_element(arr.begin(), arr.end());
    sum1 -= *itMax;
    sum2 += *itMax;
    arr.erase(itMax);
  }

  LOG(size - arr.size());
  return 0;
}