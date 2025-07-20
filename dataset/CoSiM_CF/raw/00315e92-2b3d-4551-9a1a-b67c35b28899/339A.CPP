#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("c.in", "rt", stdin);
#endif // ONLINE_JUDGE

  string s;
  vector<int> arr;

  cin >> s;

  int n = s.length();

  for (int c = 0; c < n; c++) {
    if (s[c] != '+') {
      arr.push_back(s[c]);
    }
  }

  sort(arr.begin(), arr.end());

  for (int i = 0; i < arr.size(); i++) {
    if (i != arr.size() - 1)
      cout << (char)arr[i] << "+";
    else
      cout << (char)arr[i];
  }

  return 0;
}
