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
  s[0] = (s[0] > 90) ? (s[0] - 32) : s[0];
  cout << s;
  return 0;
}