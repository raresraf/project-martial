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
  int numLucky = 0;
  for (int i = 0; i < s.size(); i++) {
    numLucky += (s[i] == '4' || s[i] == '7');
  }
  cout << ((numLucky == 4 || numLucky == 7) ? "YES" : "NO");
  return 0;
}