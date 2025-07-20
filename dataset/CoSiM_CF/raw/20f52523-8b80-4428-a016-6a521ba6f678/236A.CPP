#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;

int main() {
  string s;
  int count = 0;
  vector<bool> memo(26, false);
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (memo[s[i] - 'a'] == true)
      continue;
    count++;
    memo[s[i] - 'a'] = true;
  }
  if (count % 2)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;
  return 0;
}
