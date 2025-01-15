#include <bits/stdc++.h>

using namespace std;

int main() {
  vector<int> vec;
  string s;
  string tmp;
  cin >> s;
  int len = s.size();
  int left = 0;

  while (left < len) {
    if (s[left] != '+') {
      char p[2] = {s[left], 0};
      string t = p;
      tmp += t;
      left++;
    } else {
      int tt = stoi(tmp);
      vec.push_back(tt);
      tmp = "";
      left++;
    }
  }
  char t1[2] = {s[s.size() - 1], 0};
  string sss = t1;

  int ttt = stoi(sss);
  vec.push_back(ttt);
  string ret = "";
  sort(vec.begin(), vec.end());
  for (auto i : vec) {
    ret += to_string(i) + '+';
  }
  ret = ret.substr(0, ret.size() - 1);
  cout << ret;
  return 0;
}