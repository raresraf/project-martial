#include <bits/stdc++.h>
using namespace std;

int main(void) {
  vector<int> arr;
  int arraySize, k, a;

  cin >> arraySize >> k;

  while (arraySize--) {
    cin >> a;

    arr.push_back(a);
  }

  a = arr[k - 1];
  sort(arr.begin(), arr.end());
  k = 0;

  for (auto i = arr.end() - 1; i >= arr.begin(); --i) {
    if (*i >= a && *i > 0) {
      k++;
    } else {
      break;
    }
  }

  cout << k;

  return 0;
}
