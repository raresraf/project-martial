// http://codeforces.com/problemset/problem/266/A
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string input;
  cin >> n;
  cin >> input;
  vector<char> stones;
  for (int i = 0; i < n; i++)
    stones.push_back(input[i]);
  int removals = 0;
  for (auto itr = stones.begin(); itr < stones.end() - 1; itr++) {
    if (*itr == *(itr + 1)) {
      stones.erase(itr + 1);
      removals++;
    }
  }
  cout << removals << endl;
  return 0;
}
