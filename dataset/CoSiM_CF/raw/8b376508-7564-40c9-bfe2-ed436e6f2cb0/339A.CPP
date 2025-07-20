// Problem Link https://codeforces.com/problemset/problem/339/A

#include <iostream>

using namespace std;

int main() {
  string sum;
  cin >> sum;

  int nums[3];
  nums[0] = nums[1] = nums[2] = 0;
  int num;
  for (int i = 0; i < sum.length(); i += 2) {
    num = sum[i] - '0';
    nums[num - 1]++;
  }

  bool first = true;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < nums[i]; j++) {
      if (!first) {
        cout << "+";
      }
      cout << i + 1;
      first = false;
    }
  }
  cout << "\n";
}