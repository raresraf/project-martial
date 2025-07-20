#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int i;
  string out, str;
  vector<int> num;
  cin >> str;
  for (i = 0; i < str.size(); i++) {
    if (isdigit(str[i]))
      num.push_back(str[i]);
  }
  sort(num.begin(), num.end());
  out += num[0];
  for (i = 1; i < num.size(); i++) {
    out += "+";
    out += num[i];
  }
  out.erase(out.size());
  cout << out << endl;
  return 0;
}
