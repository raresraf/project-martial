#include <bits/stdc++.h>

using namespace std;
string helpful_math(char *str) {
  vector<string> ret;
  string s = "";
  while (*str) {
    if (*str != 43) {
      string tmp;
      tmp = *str;
      ret.push_back(tmp);
      *str++;
    } else
      *str++;
  }
  sort(ret.begin(), ret.end());
  for (auto i : ret)
    s += i + "+";
  return s;
}
int main() {
  char *p = (char *)malloc(sizeof(char) * 150);
  cin >> p;
  string s = helpful_math(p);
  s = s.substr(0, s.size() - 1);
  cout << s;
  return 0;
}