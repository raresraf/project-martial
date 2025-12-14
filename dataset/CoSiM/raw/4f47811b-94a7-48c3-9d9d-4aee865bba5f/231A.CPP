#include <iostream>

using namespace std;

int main() {
  int n = 0;
  cin >> n;
  int counter = 0;

  for (int i = 0; i < n; i++) {
    int nums[3];

    cin >> nums[0];
    cin >> nums[1];
    cin >> nums[2];

    if (nums[0] + nums[1] + nums[2] >= 2) {
      counter++;
    }
  }
  cout << counter;
  return 0;
}
