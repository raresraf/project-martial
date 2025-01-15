#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  int n;
  int total = 0;
  vector<int> nums;
  cin >> n;

  for (int i = 0; i < n; i++) {
    int num;
    cin >> num;
    nums.push_back(num);
  }

  sort(nums.begin(), nums.end(), greater<int>());

  for (int x : nums)
    total += x;
  int j;
  for (j = 0; j < nums.size();) {
    int minsum = 0;
    int remaining = total;
    while (!(minsum > remaining)) {
      minsum += nums.at(j);
      remaining -= nums.at(j);
      j++;
    }
    break;
  }

  cout << j << endl;

  return 0;
}
