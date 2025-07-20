#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace std;

int main() {
  string s;
  cin >> s;
  set<char> seen;
  for (char c : s) {
    seen.insert(c);
  }
  cout << ((seen.size() % 2 == 0) ? "CHAT WITH HER!" : "IGNORE HIM!");
  return 0;
}