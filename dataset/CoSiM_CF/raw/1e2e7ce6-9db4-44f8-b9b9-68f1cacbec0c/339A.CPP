#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  string out = "";
  vector<int> nums;

  for (char c : str) {
    int ascii = (int)c;
    if (ascii != 43) {
      nums.push_back(ascii - 48);
    }
  }
  sort(nums.begin(), nums.end());

  vector<int>::iterator it;
  for (it = nums.begin(); it != nums.end(); ++it) {
    out += to_string(*it);
    out += "+";
  }

  cout << out.substr(0, out.size() - 1) << endl;

  return 0;
}
