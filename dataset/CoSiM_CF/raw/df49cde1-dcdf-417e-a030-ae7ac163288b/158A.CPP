#include <bits/stdc++.h>

using namespace std;

int main() {
  int count;
  cin >> count;
  vector<string> vec;
  while (count-- > 0) {
    string s;
    cin >> s;
    vec.push_back(s);
  }
  for (auto &i : vec) {

    if (i.size() > 10) {
      string pls;
      string c;
      pls = to_string(i.size() - 2);
      string ret = i[0] + pls + i[i.size() - 1];
      cout << ret << endl;
    } else
      cout << i << endl;
  }
  return 0;
}
