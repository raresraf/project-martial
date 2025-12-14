/**
 * I'm sorry I was an Idiot
 */
#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int nums[n];
  for (int i = 0; i < n; ++i) {
    cin >> nums[i];
  }
  if (nums[k - 1] > 0) {
    int m = k;
    while (m < n && nums[m] == nums[k - 1]) {
      ++m;
    }
    cout << m << endl;
  } else {
    int m = k - 2;
    while (m >= 0 && nums[m] <= 0) {
      --m;
    }
    cout << m + 1 << endl;
  }
  return 0;
}