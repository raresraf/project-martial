#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int w{0};
  vector<string> v;
  cin >> w;
  for (int i{0}; i < w; i++) {
    string s;
    cin >> s;
    v.push_back(s);
  }
  for (auto s : v) {
    if (s.length() > 10) {
      cout << s.at(0) << s.substr(1, s.length() - 2).length()
           << s.at(s.length() - 1) << endl;
    } else
      cout << s << endl;
  }
  return 0;
}
