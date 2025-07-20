#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  s.erase(remove(s.begin(), s.end(), '+'), s.end());
  sort(s.begin(), s.end());

  for (auto i = 0; i != s.size() - 1; i += 2) {
    s.insert(i + 1, "+");
  }
  cout << s;
}