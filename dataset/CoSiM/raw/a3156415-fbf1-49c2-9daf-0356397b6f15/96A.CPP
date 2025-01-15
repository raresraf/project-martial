#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;

int main() {
  string s;
  bool flag = false;
  cin >> s;
  pair<int, int> p = make_pair(s[0], 1);
  for (int i = 1; i < s.size(); i++) {
    if (p.first == s[i]) {
      if (p.second + 1 == 7) {
        flag = true;
        break;
      } else
        p.second += 1;
    } else {
      p.first = s[i];
      p.second = 1;
    }
  }
  if (flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
