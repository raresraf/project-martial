// https://codeforces.com/problemset/problem/339/A
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

vector<int> getNuums(string s) {
  vector<int> nums;
  char delimiter = '+';
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == delimiter) {
      continue;
    }
    int x = s.at(i) - 48;
    nums.push_back(x);
  }
  sort(nums.begin(), nums.end());
  return nums;
}
int main() {
  string r;
  string s;
  cin >> s;
  vector<int> nums = getNuums(s);
  for (auto i : nums) {
    r += to_string(i);
    r += '+';
  }
  r = r.substr(0, r.size() - 1);
  cout << r << endl;
}